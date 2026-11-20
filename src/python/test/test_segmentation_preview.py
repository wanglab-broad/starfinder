"""The preview of input recipes, ``starfinder segment preview``: rows PV1 to PV4 (W-891).

design/validation/segmentation-input-recipe.md, "The preview tool (PV)". The fixture is a workflow configuration
of two FOVs (8 × 32 × 32, uint16) whose saved forms the workflow would read: the reference image that
``FOV.save_reference_image`` writes and the morphology round registered by ``FOV.register`` with checkpoints
(``other_rounds/morph/``), the voxel sizes declared, and a nucleus mask imported from the input root. The run
``cell`` grows cells by ``seeded_watershed`` from the imported nuclei; the run ``nucleus_sd`` is StarDist on the
library stub of ``test_segmentation_recipe.py`` (no model is loaded). Nothing is written outside the preview's
folder.
"""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from starfinder.dataset import CheckpointConfig
from starfinder.export._record import directory_files
from starfinder.image import ImageMetadata
from starfinder.io import load_volume, save_volume
from starfinder.segmentation import (InputChannel, LabelImportConfig, SeededWatershedConfig,
                                     SegmentationPlan, SegmentationRun, apply_input_recipe, preview_recipes)
from starfinder.segmentation._labels import array_sha256, grid_sha256

from .test_registration_other_rounds import SHIFT, TRANSLATION, grid, texture
from .test_segmentation_recipe import stardist_folder, stubs  # noqa: F401  (the library stubs of MI3)

pytestmark = [pytest.mark.segmentation, pytest.mark.validation, pytest.mark.contract]

F32 = {"rtol": 1e-6, "atol": 1e-6}
F64 = {"rtol": 1e-9, "atol": 1e-12}
FOVS = ("FOV_001", "FOV_002")
SPACING = (0.35, 0.1, 0.1)
MORPH = [{"channel": "ch00", "name": "DAPI"}, {"channel": "ch01", "name": "Nissl"}, {"channel": "ch02", "name": "A"},
         {"channel": "ch03", "name": "B"}]
RECIPES = {"plain": {}, "smooth": {"channels": [{"method": "gaussian", "sigma_um": 0.2}]},
           "sd_plain": {"model_input": {"factor": 1.0}},
           "sd_smooth": {"after_projection": [{"method": "gaussian", "sigma_um": 0.2}],
                         "model_input": {"pixel_size_um": 0.2}}}


def configuration(work, model):
    return {"backend": "python", "root_input_path": str(work / "input"), "dataset_id": "data", "sample_id": "sample",
            "root_output_path": str(work / "output"), "output_id": "out", "n_rounds": 2, "ref_round": "round1",
            "rotate_angle": 0, "seq_channel_order": ["ch01", "ch02", "ch03", "ch04"],
            "additional_round": [{"round_name": "morph", "channel_order": MORPH}],
            "voxel_size_xy": SPACING[1], "voxel_size_z": SPACING[0], "morphology": {},
            "segmentation": {"input_recipes": RECIPES, "runs": [
                {"name": "nucleus", "target": "nucleus", "method": "import", "path": "labels/{fov_id}.tif"},
                {"name": "cell", "target": "cell", "method": "seeded_watershed", "sigma_um": 0.1, "seeds": "nucleus",
                 "inputs": [{"role": "cytoplasm", "round": "morph", "channel": "Nissl"}], "input_recipe": "plain"},
                {"name": "nucleus_sd", "target": "nucleus", "method": "stardist", "model_path": str(model),
                 "inputs": [{"role": "nuclear", "round": "morph", "channel": "DAPI"}], "input_recipe": "sd_plain"}]}}


def write_fov(dataset, fov_id):
    """One FOV's saved forms: the reference image and the registered morphology round, and its nucleus mask."""
    p = grid()
    fov = dataset.fov(fov_id)
    fov.images = {"round1": np.stack([texture(s, p) for s in (1, 2, 3, 0)], axis=-1),
                  "round2": np.stack([texture(s, p) for s in (6, 7, 8, 9)], axis=-1),
                  "morph": np.stack([texture(s, p - SHIFT.reshape(3, 1, 1, 1)) for s in (0, 4, 5, 10)], axis=-1)}
    fov.metadata = {name: ImageMetadata(f"{fov_id}/{name}", SPACING, spatial_unit="micrometer") for name in fov.images}
    fov.register(TRANSLATION, rounds=["morph"], checkpoints=CheckpointConfig())
    fov.save_reference_image()
    labels = np.zeros(fov.reference_grid().shape_zyx, np.uint16)
    labels[:, 4:10, 4:11], labels[:, 18:25, 15:22] = 3, 8
    save_volume(labels, Path(dataset.input_root) / "labels" / f"{fov_id}.tif", metadata=fov.reference_grid().metadata)


@pytest.fixture
def dataset_config(tmp_path):
    from starfinder.dataset.workflow import _workflow_dataset
    from starfinder.workflow._plan import resolve
    from starfinder.workflow._translate import _base
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    work = tmp_path / "work"
    config = configuration(work, stardist_folder(tmp_path, 2))
    path = work / "starfinder.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    dataset = _workflow_dataset(_base(resolve(config, {"path": str(path), "sha256": None})["configuration"]))
    for fov_id in FOVS:
        write_fov(dataset, fov_id)
    return path, dataset


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def segmented(dataset, fov_id, recipe_name, run="cell"):
    """FOV.segment of the run with the recipe, from the saved forms as the preview reads them."""
    from starfinder.dataset.workflow import _input_recipe
    fov = dataset.fov(fov_id).load_reference_image()
    fov.load_registered_round("morph")
    recipe = _input_recipe(RECIPES[recipe_name], recipe_name, recipe_name)
    nucleus = SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(
        str(Path(dataset.input_root) / "labels" / f"{fov_id}.tif"), "nucleus"))
    cell = SegmentationRun("cell", "cell", (InputChannel("cytoplasm", "morph", "Nissl"),), SeededWatershedConfig(0.1),
                           seeds="nucleus", input_recipe=recipe)
    from starfinder.segmentation._plan import segment_fov
    inputs = {}
    results = segment_fov(fov, SegmentationPlan((nucleus, cell)), inputs=inputs)
    return results[run], inputs[run]


# --- PV1: the outputs ------------------------------------------------------------------------------------------------

def test_pv1_the_outputs(dataset_config, tmp_path):
    path, dataset = dataset_config
    output = tmp_path / "preview"
    record = preview_recipes(path, run="cell", fovs=FOVS, recipes=["plain", "smooth"], output=output)
    assert sorted(p.name for p in output.iterdir()) == ["overlay.png", "preview.json", "sizes.csv", "sizes.png",
                                                        "summary.csv"]
    from PIL import Image
    with Image.open(output / "overlay.png") as image:
        width, height = image.size
    assert width == int(3.2 * 2 * 100) and height == int(3.2 * 2 * 2 * 100)   # 2 FOVs × (2 rows) × 2 recipes
    sizes, summary = read_csv(output / "sizes.csv"), read_csv(output / "summary.csv")
    assert len(summary) == 4 and json.loads((output / "preview.json").read_text()) == record
    pixel = SPACING[1] * SPACING[2]
    for fov_id in FOVS:
        for name in ("plain", "smooth"):
            result, _ = segmented(dataset, fov_id, name)
            labels = result.labels
            rows = [r for r in sizes if r["fov"] == fov_id and r["recipe"] == name]
            values = sorted(set(np.unique(labels).tolist()) - {0})
            assert [int(r["label"]) for r in rows] == values
            for row in rows:
                footprint = np.any(labels == int(row["label"]), axis=0).sum()
                assert float(row["area_um2"]) == pytest.approx(footprint * pixel, rel=1e-9, abs=1e-12)
                assert float(row["equivalent_diameter_um"]) == pytest.approx(
                    2 * np.sqrt(float(row["area_um2"]) / np.pi), rel=1e-9, abs=1e-12)
            (line,) = [r for r in summary if r["fov"] == fov_id and r["recipe"] == name]
            diameters = np.array([float(r["equivalent_diameter_um"]) for r in rows])
            assert int(line["n_labels"]) == len(rows)
            for p in (5, 25, 50, 75, 95):
                assert float(line[f"diameter_p{p}_um"]) == pytest.approx(np.percentile(diameters, p), rel=1e-9,
                                                                         abs=1e-12)
    assert record["grid"] == "fov" and record["run"] == "cell" and len(record["results"]) == 4
    assert set(record["files"]) == {"overlay.png", "sizes.png", "sizes.csv", "summary.csv"}


# --- PV2: the same as the routes -------------------------------------------------------------------------------------

def test_pv2_the_labels_and_the_model_input_are_those_of_fov_segment(dataset_config, tmp_path):
    path, dataset = dataset_config
    record = preview_recipes(path, run="cell", fovs=FOVS, recipes=["plain", "smooth"], output=tmp_path / "preview",
                             save_inputs=True)
    for entry in record["results"]:
        result, segmentation_input = segmented(dataset, entry["fov"], entry["recipe"])
        assert entry["labels"]["sha256"] == array_sha256(result.labels)
        assert entry["model_input"]["sha256"] == array_sha256(segmentation_input.image)
        assert entry["recipe_record"]["sha256"] == result.record["input"]["recipe"]["sha256"]
        saved = load_volume(tmp_path / "preview" / "inputs" / entry["fov"] / f"{entry['recipe']}.ome.tif").image
        assert np.array_equal(saved.reshape(segmentation_input.image.shape), segmentation_input.image)


@pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                    reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed")
def test_pv2_the_fused_view_is_the_recipe_on_one_view_at_k(dataset_config, tmp_path, stubs):
    from starfinder.export._fuse import fuse_one_view
    from starfinder.dataset.workflow import _input_recipe
    from starfinder.segmentation import ReferenceGrid
    path, dataset = dataset_config
    record = preview_recipes(path, run="nucleus_sd", fovs=FOVS[:1], recipes=["sd_plain", "sd_smooth"],
                             output=tmp_path / "preview", factor_yx=2, save_inputs=True)
    assert record["grid"] == "fused_view" and record["factor_yx"] == 2
    fov = dataset.fov(FOVS[0]).load_reference_image()
    fov.load_registered_round("morph")
    dapi = np.asarray(fov.images["morph"])[..., 0]
    fused, metadata = fuse_one_view(dapi, SPACING, 2, name=FOVS[0])
    view = ReferenceGrid(fused.shape, metadata, "declared")
    for entry in record["results"]:
        recipe = _input_recipe(RECIPES[entry["recipe"]], entry["recipe"], entry["recipe"])
        expected, _, recipe_record = apply_input_recipe([("nuclear", fused, {})], view, recipe=recipe, k=2,
                                                        fit_mask=np.ones(view.shape_zyx, bool), run="nucleus_sd",
                                                        projected="fused_maximum")
        saved = load_volume(tmp_path / "preview" / "inputs" / FOVS[0] / f"{entry['recipe']}.ome.tif").image
        np.testing.assert_allclose(saved.reshape(expected.image.shape), expected.image, **F32)
        np.testing.assert_allclose(entry["recipe_record"]["model_input"]["factor_zyx"],
                                   recipe_record["model_input"]["factor_zyx"], **F64)
    np.testing.assert_allclose(record["results"][1]["recipe_record"]["model_input"]["factor_zyx"],
                               [1.0, 2 * SPACING[1] / 0.2, 2 * SPACING[1] / 0.2], **F64)


# --- PV2: the export run on the raw reference stain, and the label operations on every path (W-907) ---------------

needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
STAIN = "ch05"
# In pixels: a projected run grid has no spacing (ImageMetadata.projected), and at K = 1 a pixel of the fused view
# is a pixel of the FOV.
EXPAND = {"operation": "expand_labels", "distance": 2, "unit": "pixel", "mode": "planar"}
MAX_RECIPE = {"projection": {"method": "max"}, "model_input": {"factor": 1.0}}


# How the reference stain is there: only its raw files, with a morphology block and the other round morph
# (``prepared``) or without either (``no_morphology_block``, the configuration of W-891's review probe), or also its
# saved form, which FOV.prepare_morphology wrote (``saved``). No form has a saved reference image.
STAIN_FORMS = ("prepared", "no_morphology_block", "saved")


def stain_dataset(tmp_path, form="prepared"):
    """Two FOVs with the raw files of the reference round's stain and no saved reference image (``STAIN_FORMS``). The
    export run ``nucleus_fused`` reads the image ``reference_stain`` of export.images, and the run ``nucleus_grown``
    of segmentation.runs grows its labels by ``expand_labels``."""
    from starfinder.dataset import MorphologyConfig
    from starfinder.dataset.workflow import _workflow_dataset
    from starfinder.workflow._plan import resolve
    from starfinder.workflow._translate import _base
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    work = tmp_path / "work"
    model = stardist_folder(tmp_path, 2)
    config = configuration(work, model)
    config["reference_stains"] = [{"channel": STAIN, "name": "PI"}]
    config["segmentation"]["input_recipes"]["sd_max"] = MAX_RECIPE
    config["segmentation"]["runs"] = [
        {"name": "nucleus_grown", "target": "nucleus", "method": "stardist", "model_path": str(model),
         "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "PI"}], "input_recipe": "sd_max",
         "operations": [dict(EXPAND)]}]
    config["stitching"] = {"method": "multiview_stitcher", "input": "reference_stain",
                           "positions": {"source": "regular_grid",
                                         "regular_grid": {"columns": 2, "rows": 1,
                                                          "order": "column_major_down_right", "overlap": 0.25}}}
    config["assembly"] = {"assignment": None}
    if form != "prepared":
        del config["morphology"]
        config["additional_round"] = []
    config["export"] = {"images": ["reference_stain"], "route": "after_fusion", "segmentation": {
        "device": "cpu", "cells": "nucleus_fused", "runs": [
            {"name": "nucleus_fused", "target": "nucleus", "method": "stardist", "model_path": str(model),
             "inputs": [{"role": "nuclear", "image": "reference_stain"}], "input_recipe": "sd_plain"}]}}
    path = work / "starfinder.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    dataset = _workflow_dataset(_base(resolve(config, {"path": str(path), "sha256": None})["configuration"]))
    p = grid()
    for i, fov_id in enumerate(FOVS):
        save_volume(texture(11 + i, p), Path(dataset.input_root) / "round1" / fov_id / f"{fov_id}_{STAIN}.tif",
                    metadata=ImageMetadata(f"{fov_id}/round1", SPACING, spatial_unit="micrometer"))
        if form == "saved":
            dataset.fov(fov_id).prepare_morphology(MorphologyConfig(rotation_degrees=0), checkpoints=CheckpointConfig())
    return path, dataset



@pytest.fixture
def thresholding(stubs, monkeypatch):
    """The StarDist stub with labels: the connected components of the normalized model input above 0.5."""
    from skimage.measure import label

    from .test_segmentation_recipe import CALLS, _StarDistStub

    class Thresholding(_StarDistStub):
        def predict_instances(self, x, axes, scale, **options):
            CALLS.append({"scale": scale, "axes": axes, "shape": x.shape})
            return label(np.asarray(x) > 0.5).astype(np.int32), {"prob": []}

    monkeypatch.setattr(sys.modules["stardist.models"], "StarDist2D", Thresholding)
    return CALLS


@needs_library
@pytest.mark.parametrize("form", STAIN_FORMS)
def test_pv2_an_export_run_on_the_raw_reference_stain_needs_no_saved_reference_image(tmp_path, stubs, form):
    from starfinder.dataset._export import reference_stain_path
    from starfinder.dataset.workflow import _input_recipe
    from starfinder.export._fuse import fuse_one_view
    from starfinder.segmentation import ReferenceGrid
    path, dataset = stain_dataset(tmp_path, form)
    roots = (Path(dataset.output_root), Path(dataset.input_root))
    before = [directory_files(root) for root in roots]
    for fov_id in FOVS:
        assert not dataset.fov(fov_id)._reference_image_path().exists()
        assert reference_stain_path(dataset.output_root, fov_id).exists() is (form == "saved")
    record = preview_recipes(path, run="nucleus_fused", fovs=FOVS, recipes=["sd_plain", "sd_smooth"],
                             output=tmp_path / "preview", factor_yx=2, save_inputs=True)
    assert record["route"] == "after_fusion" and record["grid"] == "fused_view" and len(record["results"]) == 4
    assert [directory_files(root) for root in roots] == before
    p = grid()
    for i, fov_id in enumerate(FOVS):
        stain = texture(11 + i, p)   # the raw file the fixture wrote, unrotated (rotate_angle 0)
        # The grid's source: the reference stain prepared in memory, the same acquisition as the reference round.
        entry = record["fov_grids"][fov_id]
        assert entry["source"] == "fov:reference_stain" and entry["shape_zyx"] == [8, 32, 32]
        assert entry["sha256"] == grid_sha256(stain[..., np.newaxis])
        np.testing.assert_allclose(entry["spacing_zyx"], SPACING, **F64)
        fused, metadata = fuse_one_view(stain, SPACING, 2, name=fov_id)
        view = ReferenceGrid(fused.shape, metadata, "declared")
        for entry in (e for e in record["results"] if e["fov"] == fov_id):
            recipe = _input_recipe(RECIPES[entry["recipe"]], entry["recipe"], entry["recipe"])
            model, _, _ = apply_input_recipe([("nuclear", fused, {})], view, recipe=recipe, k=2,
                                             fit_mask=np.ones(view.shape_zyx, bool), run="nucleus_fused",
                                             projected="fused_maximum")
            saved = load_volume(tmp_path / "preview" / "inputs" / fov_id / f"{entry['recipe']}.ome.tif").image
            np.testing.assert_allclose(saved.reshape(model.image.shape), model.image, **F32)
            assert entry["operations"] == []


@needs_library
@pytest.mark.parametrize("form", ["prepared", "saved"])
def test_pv2_a_fused_view_applies_and_records_the_label_operations(tmp_path, thresholding, form):
    path, _ = stain_dataset(tmp_path, form)
    common = dict(run="nucleus_grown", fovs=FOVS, recipes=["sd_max"])
    per_fov = preview_recipes(path, output=tmp_path / "fov", **common)
    view = preview_recipes(path, output=tmp_path / "view", factor_yx=1, save_inputs=True, **common)
    assert per_fov["grid"] == "fov" and view["grid"] == "fused_view" and view["route"] == "per_fov"
    for fov_id in FOVS:
        (fov_entry,) = [e for e in per_fov["results"] if e["fov"] == fov_id]
        (view_entry,) = [e for e in view["results"] if e["fov"] == fov_id]
        for entry in (fov_entry, view_entry):
            (operation,) = entry["operations"]
            assert operation["operation"] == "expand_labels"
            assert operation["config"] == {"distance": 2.0, "unit": "pixel", "mode": "planar"}
            assert operation["record"]["voxels_added"] > 0 and entry["labels"]["n_labels"] > 0
            assert operation["record"]["output"] == entry["labels"]["sha256"]
        # At K = 1 the fused view is the FOV's grid in Y and X, and the run's Z maximum is the view's (fuse
        # projects each FOV by its Z maximum): the labels are the same.
        assert view_entry["labels"]["sha256"] == fov_entry["labels"]["sha256"]
        # Without the operation the preview returns the method's labels, the operation's input.
        record = view_entry["operations"][0]["record"]
        assert record["inputs"][0] != record["output"]
        np.testing.assert_allclose(record["physical_distance_yx"], [2 * SPACING[1], 2 * SPACING[2]], **F64)


@needs_library
def test_pv3_a_fused_view_refuses_extension_through_z(tmp_path):
    path, _ = stain_dataset(tmp_path, "saved")
    config = yaml.safe_load(path.read_text())
    config["segmentation"]["runs"][0]["operations"] = [
        {"operation": "extend_labels_through_z", "median_um": 0.1, "min_area_um2": 0.1, "dilation_um": 0.1,
         "fill_holes": "twice"}]
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match=r"extend their labels through Z .* preview them without factor_yx"):
        preview_recipes(path, run="nucleus_grown", fovs=FOVS[:1], recipes=["sd_max"], output=tmp_path / "view",
                        factor_yx=1)
    assert not (tmp_path / "view").exists()


# --- PV3: nothing written outside --output -----------------------------------------------------------------------

def test_pv3_nothing_is_written_outside_the_output(dataset_config, tmp_path):
    path, dataset = dataset_config
    roots = (Path(dataset.output_root), Path(dataset.input_root))
    before = [directory_files(root) for root in roots]
    preview_recipes(path, run="cell", fovs=FOVS, recipes=["smooth"], output=tmp_path / "preview")
    assert [directory_files(root) for root in roots] == before
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "kept.txt").write_text("x")
    cases = [(dict(output=busy), "must be absent or empty"),
             (dict(fovs=[f"FOV_{i:03d}" for i in range(1, 10)]), "at most 8 FOVs and 8 recipes"),
             (dict(recipes=["plain"] * 9), "at most 8 FOVs and 8 recipes"),
             (dict(fovs=["FOV_003"]), "round 'morph' is read from its saved form .*: run the morphology task"),
             (dict(recipes=["rough"]), "not recipes of segmentation.input_recipes"),
             (dict(run="tissue"), "run 'tissue' is not a run of segmentation.runs")]
    for change, message in cases:
        options = dict(run="cell", fovs=FOVS, recipes=["plain"], output=tmp_path / "refused")
        options.update(change)
        with pytest.raises(ValueError, match=message):
            preview_recipes(path, **options)
        assert not (tmp_path / "refused").exists()
    assert sorted(p.name for p in busy.iterdir()) == ["kept.txt"]
    assert [directory_files(root) for root in roots] == before


def test_pv3_a_recipe_file_tries_a_recipe_before_it_is_written(dataset_config, tmp_path):
    path, _ = dataset_config
    recipes = tmp_path / "recipes.yaml"
    recipes.write_text(yaml.safe_dump({"wider": {"channels": [{"method": "gaussian", "sigma_um": 0.3}]}}))
    record = preview_recipes(path, run="cell", fovs=FOVS[:1], recipes=["wider"], recipe_file=recipes,
                             output=tmp_path / "preview", crop_um=(0.5, 0.5, 2.0, 2.0))
    assert record["recipes"]["wider"]["config"]["channels"] == [{"method": "gaussian", "sigma_um": 0.3}]
    with pytest.raises(ValueError, match="is in both segmentation.input_recipes and the recipe file"):
        recipes.write_text(yaml.safe_dump({"plain": {}}))
        preview_recipes(path, run="cell", fovs=FOVS[:1], recipes=["plain"], recipe_file=recipes,
                        output=tmp_path / "again")


# --- PV4: the command ----------------------------------------------------------------------------------------------

OPTIONS = ("--run", "--fovs", "--recipes", "--output", "--recipe-file", "--factor-yx", "--crop-um", "--device",
           "--save-inputs")


def test_pv4_the_command_lists_the_options_and_runs(dataset_config, tmp_path):
    path, _ = dataset_config
    done = subprocess.run([sys.executable, "-m", "starfinder", "segment", "preview", "--help"], capture_output=True,
                          text=True, check=True, cwd=Path(__file__).resolve().parents[1])
    for option in OPTIONS:
        assert option in done.stdout, option
    assert "CONFIG" in done.stdout.upper()
    from starfinder.__main__ import main
    assert main(["segment", "preview", str(path), "--run", "cell", "--fovs", "FOV_001", "--recipes", "plain,smooth",
                 "--output", str(tmp_path / "cli")]) == 0
    assert (tmp_path / "cli" / "preview.json").is_file()
    with pytest.raises(SystemExit) as raised:
        main(["segment", "preview", str(path), "--run", "cell", "--fovs", "FOV_001", "--recipes", "rough",
              "--output", str(tmp_path / "cli2")])
    assert raised.value.code == 2


def test_pv4_the_cli_reference_is_generated_from_the_parser():
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "docs" / "_ext"))
    try:
        import experiment_commands
        commands, arguments = experiment_commands.command_rows(experiment_commands.SEGMENT_COMMANDS)
    finally:
        sys.path.remove(str(root / "docs" / "_ext"))
    assert [row[0] for row in commands] == ["starfinder segment preview"]
    assert [row[0] for row in arguments] == ["CONFIG", *OPTIONS]
    from . import docs_pages
    assert "```{segment-commands}" in docs_pages.read("segmentation-algorithms.md")
