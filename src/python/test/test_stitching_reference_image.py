"""W-397: where a FOV's reference image is can be configured (``reference_image``), and the stitching
entries take the grid of a file without stored calibration from the configuration.

Criterion 1 (the configured reference image) and criterion 2 (the stitching entries and the
declared grid) of W-397; the tests of the workflow's stitching adapters moved to the stitching commands
(test_workflow_stitch_tasks.py, W-750), and the rule inputs of the reference image to the plan's
declarations (test_workflow_plan.py, test_workflow_plan_refusals.py). Fixtures, all written in the test:

* ``scene_layouts``: the synthetic sample scene ``sample-tiny-v1`` (2 × 2 FOVs of 5×36×36) written
  as a dataset by ``test_stitching_declared_grids.scene_dataset``, with stored metadata and as plain
  pages with the voxel sizes, each in the default layout (``fovs/<fov>/reference.tif``) and in a
  copy whose reference images are moved to ``images/flamingo/DAPI/<fov>_reference.tif`` (another
  folder and name), with ``reference_image`` set to that pattern. Every output root follows the
  workflow's layout ``<root>/data/output``.

``Dataset.stitch`` runs multiview-stitcher (a method other than BigStitcher) on the scene. dask runs
with the synchronous scheduler.
"""
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
from starfinder.export import ExportConfig, FusionConfig, check_export
from starfinder.stitching import MultiviewStitcherConfig, positions_from_table

from .test_stitching_bigstitcher import GOLDEN_SHAPE, GOLDEN_SPACING  # noqa: F401  (test_workflow_stitch_tasks.py)
from .test_stitching_declared_grids import (
    SAMPLE,
    compare_tables,
    declared,
    scene_dataset,
)

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
            (moved_root / "fovs" / fov / "reference.tif").rename(target)
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
    assert dataset.reference_image == REFERENCE_IMAGE == "fovs/{fov_id}/reference.tif"
    assert source_path(dataset, "Position001", "reference_merged") == (tmp_path / "out" / "fovs" / "Position001" /
                                                                        "reference.tif")
    absolute = str(tmp_path / "elsewhere" / "{fov_id}.tif")
    assert reference_image_path(tmp_path / "out", "F1", absolute) == tmp_path / "elsewhere" / "F1.tif"
    assert reference_image_path(tmp_path / "out", "F1", MOVED) == tmp_path / "out" / "images/flamingo/DAPI/F1_reference.tif"
    config = {"dataset_id": "d", "output_id": "o", "sample_id": "s", "root_input_path": str(tmp_path),
              "root_output_path": str(tmp_path), "n_rounds": 1, "ref_round": "round1"}
    assert workflow._workflow_dataset(config).reference_image == REFERENCE_IMAGE
    assert workflow._workflow_dataset(dict(config, reference_image=None)).reference_image == REFERENCE_IMAGE
    assert workflow._workflow_dataset(dict(config, reference_image=MOVED)).reference_image == MOVED
    # The plan's declarations of the reference image: test_workflow_plan.py,
    # test_the_reference_image_is_declared_where_reference_image_names_it (W-750).


@pytest.mark.parametrize("value", ["fovs/{fovID}/reference.tif", "images/x.tif", "", None, 3])
def test_a_pattern_without_fov_id_raises_naming_the_setting(tmp_path, value):
    with pytest.raises(ValueError, match=r"reference_image must be a file pattern with \{fov_id\}") as error:
        Dataset(tmp_path, tmp_path, "d", "s", "o", reference_image=value)
    assert repr(value) in str(error.value)
    config = {"dataset_id": "d", "output_id": "o", "sample_id": "s", "root_input_path": str(tmp_path),
              "root_output_path": str(tmp_path), "n_rounds": 1, "ref_round": "round1", "reference_image": value}
    if value is not None:
        with pytest.raises(ValueError, match="reference_image"):
            workflow._workflow_dataset(config)
    # The plan refuses it: test_workflow_plan_refusals.py, test_a_reference_image_without_fov_id_is_refused (W-750).


def test_without_the_setting_the_default_path_is_read(scene_layouts, monkeypatch):
    """The default layout reads fovs/<fov>/reference.tif and nothing else."""
    import starfinder.io as io
    real, opened = io.load_volume, []
    monkeypatch.setattr(io, "load_volume", lambda path, *a, **k: opened.append(Path(path)) or real(path, *a, **k))
    dataset = scene_layouts[("stored", "default")]
    for fov in fov_ids(scene_layouts):
        path = dataset.output_root / "fovs" / fov / "reference.tif"
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


# --- Criterion 2: the stitching entries and the declared grid -------------------------------------------------------

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
