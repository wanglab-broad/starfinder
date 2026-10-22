"""W-397: where a FOV's reference image is can be configured (``reference_image``), and the stitching
entries take the grid of a file without stored calibration from the configuration.

Criterion 1 (the configured reference image) and criterion 2 (the stitching adapters and the
declared grid) of W-397. Fixtures, all written in the test:

* ``scene_layouts``: the synthetic sample scene ``sample-tiny-v1`` (2 × 2 FOVs of 5×36×36) written
  as a dataset by ``test_stitching_declared_grids.scene_dataset``, with stored metadata and as plain
  pages with the voxel sizes, each in the default layout (``images/ref_merged/<fov>.tif``) and in a
  copy whose reference images are moved to ``images/flamingo/DAPI/<fov>_reference.tif`` (another
  folder and name), with ``reference_image`` set to that pattern. Every output root follows the
  workflow's layout ``<root>/data/output``.
* The fixture of ``test_workflow_stitching_golden.py`` (its stage file of 8 positions on a 3 × 3 grid
  and its BigStitcher ``dataset.xml`` with ViewSetups, ``golden_stitching_setups``) written as a
  dataset of one-plane reference images of 80 × 100, with stored metadata and as plain pages with
  the voxel sizes in the configuration (0.19 and 0.34 µm).

``Dataset.stitch`` and the workflow's ``stitch_sample`` adapter run multiview-stitcher (a method
other than BigStitcher) on the scene; the BigStitcher import adapter runs on the golden fixture's
hand-built ``dataset.xml``. dask runs with the synchronous scheduler.
"""
import copy
import hashlib
import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from pandas.testing import assert_frame_equal

import starfinder.dataset.workflow as workflow
from starfinder._layout import REFERENCE_IMAGE, reference_image_path
from starfinder.assembly import load_assembly
from starfinder.dataset import Dataset
from starfinder.dataset._export import _reference_grid, source_path
from starfinder.io import save_volume
from starfinder.sample_output import ExportConfig, FusionConfig, check_export
from starfinder.stitching import MultiviewStitcherConfig, load_placement, positions_from_table

from .stitching_fixtures import LN_SPACING, grid
from .test_stitching_bigstitcher import (  # noqa: F401
    GOLDEN_SHAPE,
    GOLDEN_SPACING,
    LINK_IDS,
    LINK_SHAPE,
    golden_stitching_setups,
    links_view_map,
    links_xml,
)
from .test_stitching_declared_grids import (
    SAMPLE,
    compare_tables,
    declared,
    plain_pages,
    scene_dataset,
)
from .test_stitching_workflow import stub
from .test_workflow_stitching_golden import stage_file

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
MOVED = "images/flamingo/DAPI/{fov_id}_reference.tif"
MULTIVIEW = MultiviewStitcherConfig(binning_zyx=(1, 2, 2), quality_threshold=-1.0)
EXPORT = ExportConfig(fusion=FusionConfig(images=("reference_merged",), dimensionality="both"))


@pytest.fixture(scope="module")
def scene_layouts(tmp_path_factory):
    """Datasets keyed (stored or plain, default or moved); the moved copies read their images where they lie."""
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    root = tmp_path_factory.mktemp("w397_scene")
    saved = save_sample_scene(scene, root / "scene")
    out = {"scene": scene, "saved": saved}
    for form in ("stored", "plain"):
        dataset = scene_dataset(root / form / "data", scene, book, saved, stored=form == "stored")
        out[(form, "default")] = dataset
        moved_root = root / f"{form}_moved" / "data" / "output"
        shutil.copytree(dataset.output_root, moved_root)
        for fov in scene.positions.fov_id:
            target = reference_image_path(moved_root, fov, MOVED)
            target.parent.mkdir(parents=True, exist_ok=True)
            (moved_root / "images" / "ref_merged" / f"{fov}.tif").rename(target)
        shutil.rmtree(moved_root / "images" / "ref_merged")
        out[(form, "moved")] = replace(dataset, input_root=moved_root.parent / "input", output_root=moved_root,
                                       reference_image=MOVED)
    return out


def fov_ids(layouts):
    return list(layouts["scene"].positions.fov_id)


def table(layouts):
    return positions_from_table(layouts["saved"] / "positions.csv")


def assert_same_placement(a, b):
    """Equal rows, statuses and links; origins within 1e-9 µm."""
    origins = [f"origin_{axis}" for axis in "zyx"]
    assert list(a.fovs.fov_id) == list(b.fovs.fov_id) and list(a.fovs.status) == list(b.fovs.status)
    np.testing.assert_allclose(a.fovs[origins].to_numpy(float), b.fovs[origins].to_numpy(float), rtol=0, atol=1e-9)
    assert_frame_equal(a.fovs.drop(columns=origins), b.fovs.drop(columns=origins), check_exact=True)
    assert_frame_equal(a.links, b.links, check_exact=True)


# --- Criterion 1: the configured reference image ------------------------------------------------------------------

def test_the_setting_defaults_to_the_present_location(tmp_path):
    dataset = Dataset(tmp_path / "in", tmp_path / "out", "d", "s", "o")
    assert dataset.reference_image == REFERENCE_IMAGE == "images/ref_merged/{fov_id}.tif"
    assert source_path(dataset, "Position001", "reference_merged") == (tmp_path / "out" / "images" / "ref_merged" /
                                                                        "Position001.tif")
    absolute = str(tmp_path / "elsewhere" / "{fov_id}.tif")
    assert reference_image_path(tmp_path / "out", "F1", absolute) == tmp_path / "elsewhere" / "F1.tif"
    assert reference_image_path(tmp_path / "out", "F1", MOVED) == tmp_path / "out" / "images/flamingo/DAPI/F1_reference.tif"
    config = {"dataset_id": "d", "output_id": "o", "sample_id": "s", "root_input_path": str(tmp_path),
              "root_output_path": str(tmp_path), "n_rounds": 1, "ref_round": "round1"}
    assert workflow._workflow_dataset(config).reference_image == REFERENCE_IMAGE
    assert workflow._workflow_dataset(dict(config, reference_image=None)).reference_image == REFERENCE_IMAGE
    assert workflow._workflow_dataset(dict(config, reference_image=MOVED)).reference_image == MOVED
    assert workflow._reference_image_inputs(config, tmp_path / "d" / "o", ["{fovID}"]) == [
        str(tmp_path / "d" / "o" / "images" / "ref_merged" / "{fovID}.tif")]
    assert workflow._reference_image_inputs(dict(config, reference_image=MOVED), tmp_path, ["P1"]) == [
        str(tmp_path / "images/flamingo/DAPI/P1_reference.tif")]


@pytest.mark.parametrize("value", ["images/ref_merged/{fovID}.tif", "images/x.tif", "", None, 3])
def test_a_pattern_without_fov_id_raises_naming_the_setting(tmp_path, value):
    with pytest.raises(ValueError, match=r"reference_image must be a file pattern with \{fov_id\}") as error:
        Dataset(tmp_path, tmp_path, "d", "s", "o", reference_image=value)
    assert repr(value) in str(error.value)
    config = {"dataset_id": "d", "output_id": "o", "sample_id": "s", "root_input_path": str(tmp_path),
              "root_output_path": str(tmp_path), "n_rounds": 1, "ref_round": "round1", "reference_image": value}
    if value is not None:
        with pytest.raises(ValueError, match="reference_image"):
            workflow._workflow_dataset(config)
        with pytest.raises(ValueError, match="reference_image"):
            workflow._reference_image_inputs(config, tmp_path, ["P1"])


def test_without_the_setting_the_default_path_is_read(scene_layouts, monkeypatch):
    """The default layout reads images/ref_merged/<fov>.tif and nothing else, as before."""
    import starfinder.io as io
    real, opened = io.load_volume, []
    monkeypatch.setattr(io, "load_volume", lambda path, *a, **k: opened.append(Path(path)) or real(path, *a, **k))
    dataset = scene_layouts[("stored", "default")]
    for fov in fov_ids(scene_layouts):
        path = dataset.output_root / "images" / "ref_merged" / f"{fov}.tif"
        assert _reference_grid(dataset, fov).source == f"file:{path.resolve()}"
        assert opened[-1] == path
    moved = scene_layouts[("stored", "moved")]
    for fov in fov_ids(scene_layouts):
        path = moved.output_root / "images" / "flamingo" / "DAPI" / f"{fov}_reference.tif"
        assert _reference_grid(moved, fov).source == f"file:{path.resolve()}" and opened[-1] == path


@needs_anndata
@pytest.mark.parametrize("form", ["stored", "plain"])
def test_a_missing_file_raises_naming_the_setting_and_the_path(scene_layouts, tmp_path, form):
    """A pattern naming a file that does not exist: FileNotFoundError naming reference_image and the path; the
    export writes nothing."""
    dataset = replace(scene_layouts[(form, "moved")], reference_image="images/nowhere/{fov_id}.tif")
    missing = dataset.output_root / "images" / "nowhere" / f"{fov_ids(scene_layouts)[0]}.tif"
    expected = rf"reference image of FOV '{fov_ids(scene_layouts)[0]}' is missing: {missing} .*reference_image"
    with pytest.raises(FileNotFoundError, match=expected):
        dataset.stitch(SAMPLE, fov_ids(scene_layouts), table=table(scene_layouts), config=MULTIVIEW)
    out = tmp_path / "export"
    with pytest.raises(FileNotFoundError, match=expected):
        dataset.export(SAMPLE, placement=dataset.output_root / "stitching" / SAMPLE, config=EXPORT, output_dir=out)
    assert not out.exists()


@needs_library
@pytest.mark.slow
@pytest.mark.parametrize("form", ["stored", "plain"])
def test_the_moved_layout_gives_the_default_layouts_placement(scene_layouts, form):
    """Dataset.stitch (multiview_stitcher) with reference_image set equals the default layout without it."""
    default, moved = scene_layouts[(form, "default")], scene_layouts[(form, "moved")]
    assert default.reference_image == REFERENCE_IMAGE
    placements = [dataset.stitch(SAMPLE, fov_ids(scene_layouts), table=table(scene_layouts), config=MULTIVIEW)
                  for dataset in (default, moved)]
    assert_same_placement(*placements)
    assert set(placements[1].fovs.status) == {"image_supported"} and len(placements[1].links) > 0


@needs_library
@needs_anndata
@pytest.mark.parametrize("form", ["stored", "plain"])
def test_the_moved_layout_gives_the_default_layouts_tables_and_image(scene_layouts, tmp_path, form):
    """Dataset.export (assemble, and fuse of reference_merged: projection and stack) with reference_image set
    equals the default layout without it: the sample tables and the fused images; the fusion section names the
    moved files."""
    import dask
    default, moved = scene_layouts[(form, "default")], scene_layouts[(form, "moved")]
    records = {}
    with dask.config.set(scheduler="synchronous"):
        for key, dataset in (("default", default), ("moved", moved)):
            records[key] = dataset.export(SAMPLE, placement=dataset.output_root / "stitching" / SAMPLE,
                                          config=EXPORT, output_dir=tmp_path / key).record
            check_export(tmp_path / key)
    assert records["moved"]["parts"] == records["default"]["parts"] == {
        "assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "off", "export_spatialdata": "off"}
    compare_tables(load_assembly(tmp_path / "default"), load_assembly(tmp_path / "moved"))
    names = [o["path"] for o in records["moved"]["fusion"]["outputs"]]
    assert names == [o["path"] for o in records["default"]["fusion"]["outputs"]] == [
        "reference_merged_projection.ome.tif", "reference_merged_stack.ome.tif"]
    for name in names:
        a, b = tifffile.imread(tmp_path / "default" / name), tifffile.imread(tmp_path / "moved" / name)
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    for entry in records["moved"]["fusion"]["fovs"]:
        path = reference_image_path(moved.output_root, entry["fov_id"], MOVED)
        assert entry["images"]["reference_merged"]["path"] == str(path.resolve())
        assert entry["grid"]["source"] == ("declared" if form == "plain" else f"file:{path.resolve()}")


# --- Criterion 2: the stitching adapters and the declared grid -----------------------------------------------------

def workflow_config(dataset, **extra):
    """The workflow configuration whose output root is the dataset's (<root>/data/output)."""
    root = dataset.output_root.parent.parent
    config = {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
              "root_input_path": str(root / "input"), "root_output_path": str(root), "fov_id_pattern": "FOV_{i:03d}",
              "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 36, "img_row": 36, "rules": {}}
    if dataset.voxel_size_xy is not None:
        config.update(voxel_size_xy=dataset.voxel_size_xy, voxel_size_z=dataset.voxel_size_z)
    if dataset.reference_image != REFERENCE_IMAGE:
        config["reference_image"] = dataset.reference_image
    return config | extra


def run_workflow_stitching(dataset, scene_layouts, tmp_path):
    """stitch_sample with a multiview_stitcher block on the scene's positions table."""
    positions = {"source": "table", "table": str(scene_layouts["saved"] / "positions.csv")}
    block = {"method": "multiview_stitcher", "input": "reference_merged", "positions": positions,
             "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0}}
    config = workflow_config(dataset, stitching=block)
    sheet = tmp_path / "sheet.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(f"sample_id,fov_start,fov_end\n{SAMPLE},1,4\n")
    placement = tmp_path / "stitching" / SAMPLE / "placement.json"
    workflow._run_stitching(stub(config, inputs=[sheet], outputs=[placement], sample=SAMPLE))
    return load_placement(placement.parent)


@pytest.mark.parametrize("layout", ["default", "moved"])
def test_dataset_stitch_reads_plain_pages_on_the_declared_grid(scene_layouts, monkeypatch, layout):
    """Dataset.stitch gives the method the same images from plain pages as from stored files, on the grid the
    voxel sizes declare (the method itself is spied; the slow tests below run multiview_stitcher)."""
    import starfinder.stitching as stitching
    calls = []
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, **kwargs: calls.append(inputs) or "placed")
    for form in ("stored", "plain"):
        assert scene_layouts[(form, layout)].stitch(SAMPLE, fov_ids(scene_layouts), table=table(scene_layouts),
                                                    config=MULTIVIEW) == "placed"
    spacing = tuple(scene_layouts["scene"].fov_metadata[fov_ids(scene_layouts)[0]].spacing_zyx)
    for stored, plain in zip(*calls):
        assert plain.fov_id == stored.fov_id and np.array_equal(plain.image, stored.image)
        assert plain.grid.shape_zyx == stored.grid.shape_zyx and plain.grid.source == "declared"
        assert plain.grid.metadata == declared(plain.fov_id, spacing)
        assert plain.grid.metadata.spacing_zyx == stored.grid.metadata.spacing_zyx
        path = reference_image_path(scene_layouts[("plain", layout)].output_root, plain.fov_id,
                                    scene_layouts[("plain", layout)].reference_image)
        assert plain.source == "reference_merged" and plain.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@needs_library
@pytest.mark.slow
@pytest.mark.parametrize("layout", ["default", "moved"])
def test_dataset_stitch_on_files_without_metadata(scene_layouts, layout):
    """Plain pages with the voxel sizes give the placement of the stored copy (Dataset.stitch with
    multiview_stitcher); without the voxel sizes it raises naming them."""
    stored, plain = scene_layouts[("stored", layout)], scene_layouts[("plain", layout)]
    ids, positions = fov_ids(scene_layouts), table(scene_layouts)
    expected = stored.stitch(SAMPLE, ids, table=positions, config=MULTIVIEW)
    got = plain.stitch(SAMPLE, ids, table=positions, config=MULTIVIEW)
    assert_same_placement(expected, got)
    assert set(got.fovs.status) == {"image_supported"} and len(got.links) > 0
    unconfigured = replace(plain, voxel_size_xy=None, voxel_size_z=None)
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z: set voxel_size_xy and voxel_size_z"):
        unconfigured.stitch(SAMPLE, ids, table=positions, config=MULTIVIEW)


@needs_library
@pytest.mark.slow
@pytest.mark.parametrize("layout", ["default", "moved"])
def test_the_workflow_stitching_adapter_on_files_without_metadata(scene_layouts, tmp_path, layout):
    """stitch_sample with a multiview_stitcher block: plain pages with the configured voxel sizes give the stored
    copy's placement; without them it raises naming them."""
    stored, plain = scene_layouts[("stored", layout)], scene_layouts[("plain", layout)]
    via_workflow = {form: run_workflow_stitching(dataset, scene_layouts, tmp_path / form)
                    for form, dataset in (("stored", stored), ("plain", plain))}
    assert_same_placement(via_workflow["stored"], via_workflow["plain"])
    assert set(via_workflow["plain"].fovs.status) == {"image_supported"} and len(via_workflow["plain"].links) > 0
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z"):
        run_workflow_stitching(replace(plain, voxel_size_xy=None, voxel_size_z=None), scene_layouts,
                               tmp_path / "unconfigured")


def golden_dataset(root, form, setups):
    """The golden fixture as a workflow dataset: one-plane reference images Position1 to Position8, its stage file
    and sample annotation, and the BigStitcher folder with the hand-built dataset.xml, view map and macro."""
    xml, _, _, table, view_map, _ = setups
    out = root / "data" / "output"
    for k, fov in enumerate(table.fov_ids):
        image = (np.arange(np.prod(GOLDEN_SHAPE)).reshape(GOLDEN_SHAPE) % 251 + 3 * k).astype(np.uint16)
        path = out / "images" / "ref_merged" / f"{fov}.tif"
        if form == "stored":
            save_volume(image, path, metadata=grid(GOLDEN_SHAPE, GOLDEN_SPACING).metadata)
        else:
            plain_pages(path, image)
    (out / "documents" / "maf").mkdir(parents=True)
    (out / "documents" / "maf" / f"{SAMPLE}.maf").write_text(stage_file())
    sheet = out / "documents" / "sample-annotation.csv"
    sheet.write_text(f"sample_id,fov_start,fov_end\n{SAMPLE},1,8\n")
    folder = root / "bigstitcher" / SAMPLE
    folder.mkdir(parents=True)
    shutil.copy(xml, folder / "dataset.xml")
    view_map.to_csv(folder / "view_map.csv", index=False)
    (folder / "BigStitcher_macro.ijm").write_text("// macro\n")
    positions = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "snap_to_grid": {"overlap": 0.1}}
    config = {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
              "root_input_path": str(root / "input"), "root_output_path": str(root), "fov_id_pattern": "Position{i}",
              "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": GOLDEN_SHAPE[2],
              "img_row": GOLDEN_SHAPE[1], "rules": {},
              "stitching": {"positions": positions, "bigstitcher_save_dir": str(root / "bigstitcher")}}
    if form == "plain":
        config.update(voxel_size_xy=GOLDEN_SPACING[2], voxel_size_z=GOLDEN_SPACING[0])
    return config, sheet, folder, out


def run_import(config, sheet, folder, out, fallback=None):
    config = copy.deepcopy(config)
    config["stitching"]["fallback"] = fallback
    placement, tile_config = out / "stitching" / SAMPLE / "placement.json", out / "output" / "tile_config.csv"
    workflow._run_stitching(stub(config, inputs=[sheet, folder / "dataset.xml", folder / "view_map.csv",
                                                 folder / "BigStitcher_macro.ijm"],
                                 outputs=[placement, tile_config], sample=SAMPLE))
    return load_placement(placement.parent), tile_config.read_text()


def test_the_bigstitcher_adapters_on_the_golden_fixture_without_metadata(tmp_path, golden_stitching_setups):  # noqa: F811
    """The import adapter (with each fallback) and the macro adapter, which reads the images, on plain pages with
    the voxel sizes give the stored copy's placement, legacy tile configuration, tiles and view map; without the
    voxel sizes they raise naming them. The golden dataset.xml holds no pairwise result, so every FOV is failed
    without a fallback, as in the existing tests, and placed at its nominal position with fallback nominal."""
    results = {}
    for form in ("stored", "plain"):
        config, sheet, folder, out = golden_dataset(tmp_path / form, form, golden_stitching_setups)
        results[form] = [run_import(config, sheet, folder, out, fallback) for fallback in (None, "nominal")]
        macro_dir = tmp_path / form / "macro"
        (macro_dir / SAMPLE).mkdir(parents=True)
        macro_config = copy.deepcopy(config)
        macro_config["stitching"]["bigstitcher_save_dir"] = str(macro_dir)
        workflow._run_bigstitcher_macro(stub(macro_config, inputs=[sheet], outputs=[macro_dir / SAMPLE / "m.ijm",
                                                                                     macro_dir / SAMPLE / "v.csv"],
                                             sample=SAMPLE))
        results[form].append(macro_dir / SAMPLE)
    for (stored, stored_tiles), (plain, plain_tiles) in zip(results["stored"][:2], results["plain"][:2]):
        assert_same_placement(stored, plain)
        assert len(plain.fovs) == 8 and plain_tiles == stored_tiles
    assert set(results["plain"][0][0].fovs.status) == {"failed"}
    assert results["plain"][1][0].fovs.origin_x.notna().all()
    stored_macro, plain_macro = results["stored"][2], results["plain"][2]
    assert (plain_macro / "v.csv").read_text() == (stored_macro / "v.csv").read_text()
    assert (plain_macro / "m.ijm").read_text().replace(str(tmp_path / "plain"), "") == (
        stored_macro / "m.ijm").read_text().replace(str(tmp_path / "stored"), "")
    for tile in sorted((stored_macro / "tiles").iterdir()):
        assert np.array_equal(tifffile.imread(tile), tifffile.imread(plain_macro / "tiles" / tile.name)), tile.name
    config, sheet, folder, out = golden_dataset(tmp_path / "unconfigured", "plain", golden_stitching_setups)
    for key in ("voxel_size_xy", "voxel_size_z"):
        config.pop(key)
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z"):
        run_import(config, sheet, folder, out)
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z"):
        workflow._run_bigstitcher_macro(stub(config, inputs=[sheet], outputs=[tmp_path / "m", tmp_path / "v"],
                                             sample=SAMPLE))


def links_dataset(root, form):
    """xml_links of test_stitching_bigstitcher as a workflow dataset: four 4×16×24 reference images L0 to L3 on a
    2 × 2 grid at LN's spacing, and the BigStitcher folder with the hand-built dataset.xml, view map and macro."""
    out = root / "data" / "output"
    for k, fov in enumerate(LINK_IDS):
        image = np.full(LINK_SHAPE, k + 1, np.uint16)
        path = out / "images" / "ref_merged" / f"{fov}.tif"
        if form == "stored":
            save_volume(image, path, metadata=grid(LINK_SHAPE, LN_SPACING).metadata)
        else:
            plain_pages(path, image)
    sheet = out / "documents" / "sample-annotation.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(f"sample_id,fov_start,fov_end\n{SAMPLE},0,3\n")
    folder = root / "bigstitcher" / SAMPLE
    folder.mkdir(parents=True)
    (folder / "dataset.xml").write_text(links_xml())
    links_view_map().to_csv(folder / "view_map.csv", index=False)
    (folder / "BigStitcher_macro.ijm").write_text("// macro\n")
    positions = {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2,
                                                            "order": "column_major_down_right", "overlap": 0.25}}
    config = {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
              "root_input_path": str(root / "input"), "root_output_path": str(root), "fov_id_pattern": "L{i}",
              "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": LINK_SHAPE[2],
              "img_row": LINK_SHAPE[1], "rules": {},
              "stitching": {"positions": positions, "bigstitcher_save_dir": str(root / "bigstitcher")}}
    if form == "plain":
        config.update(voxel_size_xy=LN_SPACING[2], voxel_size_z=LN_SPACING[0])
    return config, sheet, folder, out


def test_the_import_adapter_on_the_2x2_links_fixture_without_metadata(tmp_path):
    """The hand-built xml_links dataset.xml (pairwise results for its six links): plain pages with the voxel sizes
    give the stored copy's placement and links; without the voxel sizes the adapter raises naming them."""
    (stored, _), (plain, _) = [run_import(*links_dataset(tmp_path / form, form)) for form in ("stored", "plain")]
    assert_same_placement(stored, plain)
    assert set(plain.fovs.status) == {"image_supported"} and len(plain.links) == 6
    config, sheet, folder, out = links_dataset(tmp_path / "unconfigured", "plain")
    config.pop("voxel_size_xy")
    with pytest.raises(ValueError, match="lacks voxel_size_xy: set voxel_size_xy"):
        run_import(config, sheet, folder, out)


def test_the_import_adapter_reads_the_configured_reference_image(tmp_path, golden_stitching_setups):  # noqa: F811
    """With reference_image set, the import adapter reads the moved files and gives the default layout's
    placement; the rule declares the moved files as its inputs."""
    config, sheet, folder, out = golden_dataset(tmp_path / "default", "plain", golden_stitching_setups)
    expected, _ = run_import(config, sheet, folder, out)
    config, sheet, folder, out = golden_dataset(tmp_path / "moved", "plain", golden_stitching_setups)
    config["reference_image"] = MOVED
    for fov in [f"Position{i}" for i in range(1, 9)]:
        target = reference_image_path(out, fov, MOVED)
        target.parent.mkdir(parents=True, exist_ok=True)
        (out / "images" / "ref_merged" / f"{fov}.tif").rename(target)
    got, _ = run_import(config, sheet, folder, out)
    assert_same_placement(expected, got)
    assert workflow._reference_image_inputs(config, out, ["Position1"]) == [
        str(out / "images" / "flamingo" / "DAPI" / "Position1_reference.tif")]
