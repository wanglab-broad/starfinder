"""The input recipe on the route after fusion: rows AF1 to AF8 and RE3 (W-891).

design/validation/segmentation-input-recipe.md, "The route after fusion (AF)" and "Records (RE)": the
recipe runs on the fused projections with the footprints as its fit region (AF1, AF3), one recipe means
the same µm on both routes (AF2), tiles of Z > 1 refuse what cannot be recovered from their Z maximum
(AF4), check 7 (AF5), the reuse rule (AF6), the model input file of SI1 (AF7, AF8) and its record and
checks in ``check_export`` (RE3). The fixtures are ``scene_plane`` (sample-tiny-v1, FOVs of
1 × 36 × 36), ``fused_pair`` (two FOVs of 3 × 24 × 24) and ``fused_quad_notch``
(``sample_segmentation_fixtures.py``). The rows that fuse need the ``multiview-stitcher`` extra and skip
without it; a full export also needs ``anndata``.
"""
import copy
import importlib.util
import json

import numpy as np
import pytest
import tifffile

from starfinder.assembly import AssemblyConfig, ownership_map
from starfinder.dataset import workflow
from starfinder.export import SampleSegmentationConfig, SampleSegmentationRun, check_export, segment_fused
from starfinder.export._record import directory_files
from starfinder.preprocessing import ClaheConfig, GaussianConfig, PercentileNormalizationConfig, ProjectionConfig
from starfinder.segmentation import (InputRecipe, LabelImportConfig, ModelInputConfig, RecipeStep,
                                     SeededWatershedConfig, StarDistConfig, apply_input_recipe)

from . import hash_oracles
from .sample_segmentation_fixtures import (DS, SAMPLE, route_config, scene_truth_labels,
                                           segmentation_grid, truth_import, write_folder)
from .test_segmentation_recipe import stardist_folder, stubs  # noqa: F401  (the library stubs of MI3)

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
pytestmark = [pytest.mark.stitching, pytest.mark.segmentation, pytest.mark.validation, pytest.mark.contract]

F32 = {"rtol": 1e-6, "atol": 1e-6}
F64 = {"rtol": 1e-9, "atol": 1e-12}
SMOOTH = InputRecipe(channels=(GaussianConfig(1.0),), after_projection=(PercentileNormalizationConfig(1.0, 99.0),),
                     name="smooth")


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def record_of(folder):
    return json.loads((folder / "export.json").read_text())


def nuclei(folder, grid):
    return truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))


def seeded(nuclei_path, recipe, *, image="flamingo", name="cell", extra=()):
    """The nucleus import and a seeded watershed (sigma 1 µm) cell run with ``recipe`` on one fused image."""
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei_path), "nucleus")),
            SampleSegmentationRun(name, "cell", (("cytoplasm", image),), SeededWatershedConfig(sigma_um=1.0),
                                  seeds="nucleus", input_recipe=recipe), *extra)
    return SampleSegmentationConfig(runs, name, "nucleus")


def fused(tmp_path, name, k):
    """The fixture written as a dataset and its included FOVs fused alone at ``1/k``; the folder and the grid."""
    from starfinder.export import FusionInput, fuse_images
    from starfinder.stitching._placement import OWNING

    from .sample_segmentation_fixtures import STAINS, fusion_config
    folder = write_folder(tmp_path, name)
    statuses = dict(zip(folder.placement.fovs["fov_id"], folder.placement.fovs["status"]))
    inputs = [FusionInput(f, folder.grids[f], {n: folder.image_path(n, f) for n in STAINS})
              for f in folder.scene.ids if statuses[f] in OWNING]
    fuse_images(folder.placement, inputs, config=fusion_config(k), output_dir=folder.export_dir)
    return folder, segmentation_grid(folder.export_dir)


def capture(monkeypatch):
    import starfinder.export._segment as segment_module
    captured = []
    real = segment_module.segment

    def wrapped(segmentation_input, **kwargs):
        captured.append((segmentation_input, kwargs))
        return real(segmentation_input, **kwargs)
    monkeypatch.setattr(segment_module, "segment", wrapped)
    return captured


def covered(folder, grid):
    """The pixels of the segmentation grid that ownership_map (nearest_center) gives an owner, as (1, Y, X)."""
    owners = ownership_map(folder.placement, folder.grids, config=AssemblyConfig(ownership="nearest_center"))
    (_, sy, sx), (_, oy, ox) = grid.metadata.spacing_zyx, grid.metadata.origin_zyx
    yy, xx = np.meshgrid(oy + np.arange(grid.shape_zyx[1]) * sy, ox + np.arange(grid.shape_zyx[2]) * sx,
                         indexing="ij")
    points = np.stack([np.zeros(yy.size), yy.ravel(), xx.ravel()], axis=1)
    return np.array([o is not None for o in owners.owner(points)]).reshape(1, *grid.shape_zyx[1:])


def fused_array(folder, image):
    record = record_of(folder.export_dir)
    entry = next(o for o in record["fusion"]["outputs"] if o["image"] == image and o["format"] == "ome_tiff"
                 and o["dimensionality"] == "projection")
    array = tifffile.imread(folder.export_dir / entry["path"])
    return (array[np.newaxis] if array.ndim == 2 else array), entry


def run_entry(record, name):
    return next(r for r in record["segmentation"]["runs"] if r["name"] == name)


# --- AF1: the recipe runs on the fused projection -----------------------------------------------------------------

@needs_library
@pytest.mark.parametrize("k", [1, 2])
def test_af1_the_recipe_runs_on_the_fused_projection(tmp_path, monkeypatch, k):
    folder, grid = fused(tmp_path, "scene_plane", k)
    captured = capture(monkeypatch)
    segment_fused(folder.export_dir, config=seeded(nuclei(folder, grid), SMOOTH), sample=SAMPLE,
                  identity=(DS, SAMPLE))
    (received, _), = [c for c in captured if c[1]["target"] == "cell"]
    flamingo, entry = fused_array(folder, "flamingo")
    mask = covered(folder, grid)
    expected, _, record = apply_input_recipe([("cytoplasm", flamingo, {"image": "flamingo"})], grid, recipe=SMOOTH,
                                             fit_mask=mask, k=k, run="cell", projected="single_plane")
    np.testing.assert_allclose(received.image, expected.image, **F32)
    section = record_of(folder.export_dir)["segmentation"]
    run = run_entry({"segmentation": section}, "cell")
    assert run["input"]["written"] is True and run["input"]["path"] == "model_inputs/cell.ome.tif"
    assert run["recipe"]["fit_region"] == "footprints" and run["recipe"]["fit_pixels"] == int(mask.sum())
    saved = tifffile.imread(folder.export_dir / "model_inputs" / "cell.ome.tif")
    assert saved.dtype == received.image.dtype and np.array_equal(saved, received.image[0, ..., 0])
    output = next(o for o in section["outputs"] if o.get("kind") == "model_input")
    assert output["shape_zyx"] == list(grid.shape_zyx) and [1, *saved.shape] == list(grid.shape_zyx)
    np.testing.assert_allclose(output["spacing_zyx"], grid.metadata.spacing_zyx, rtol=0, atol=1e-9)
    np.testing.assert_allclose(output["origin_zyx"], grid.metadata.origin_zyx, rtol=0, atol=1e-9)
    assert hash_oracles.file_sha256(folder.export_dir / output["path"]) == output["sha256"] == run["input"]["sha256"]


# --- AF2: one recipe, two routes, the same µm values ---------------------------------------------------------------

@needs_library
def test_af2_one_recipe_means_the_same_um_on_both_routes(tmp_path, stubs):
    from starfinder.segmentation import InputChannel, SegmentationPlan, SegmentationRun
    folder, grid = fused(tmp_path, "scene_plane", 2)
    model = StarDistConfig(model_path=str(stardist_folder(tmp_path, 2)))
    recipe = InputRecipe(channels=(GaussianConfig(1.0),), after_projection=(ClaheConfig(2.5),),
                         model_input=ModelInputConfig(pixel_size_um=2.0), name="smooth")
    fov_id = folder.scene.ids[0]
    fov = folder.dataset.fov(fov_id).load_reference_image()
    fov.segment(SegmentationPlan((SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", reference_merged=True),),
                                                  model, input_recipe=recipe),)))
    per_fov = fov.segmentation_results["nucleus"].record["input"]["recipe"]
    config = SampleSegmentationConfig((SampleSegmentationRun("nucleus", "nucleus", (("nuclear", "dapi"),), model,
                                                             input_recipe=recipe),), "nucleus")
    after = segment_fused(folder.export_dir, config=config, sample=SAMPLE, identity=(DS, SAMPLE)).record
    after = run_entry({"segmentation": after}, "nucleus")["recipe"]
    s = folder.grids[fov_id].metadata.spacing_zyx[1]
    assert grid.metadata.spacing_zyx[1] == pytest.approx(2 * s, rel=1e-9, abs=1e-12)
    for record, spacing in ((per_fov, s), (after, 2 * s)):
        (gaussian,) = [e for e in record["stages"] if e["method"] == "gaussian"]
        (clahe,) = [e for e in record["stages"] if e["method"] == "clahe"]
        np.testing.assert_allclose(gaussian["used"]["sigma_px_zyx"][1:], [1.0 / spacing] * 2, **F64)
        assert clahe["used"]["kernel_px_zyx"][1:] == [max(1, int(np.floor(2.5 / spacing + 0.5)))] * 2
        np.testing.assert_allclose(record["model_input"]["factor_zyx"], [1.0, spacing / 2.0, spacing / 2.0], **F64)
    assert per_fov["sha256"] == after["sha256"] and per_fov["name"] == after["name"] == "smooth"


# --- AF3: the fit region ---------------------------------------------------------------------------------------------

@needs_library
def test_af3_the_fit_region_is_the_footprints(tmp_path):
    folder, grid = fused(tmp_path, "fused_quad_notch", 1)
    recipe = InputRecipe(after_projection=(PercentileNormalizationConfig(5.0, 99.0),))
    result = segment_fused(folder.export_dir, config=seeded(nuclei(folder, grid), recipe), sample=SAMPLE,
                           identity=(DS, SAMPLE))
    record = run_entry({"segmentation": result.record}, "cell")["recipe"]
    mask = covered(folder, grid)
    assert mask.sum() < mask.size                                   # the placeholder leaves an L-shaped region
    assert record["fit_region"] == "footprints" and record["fit_pixels"] == int(mask.sum())
    from skimage.util import img_as_float32
    flamingo, _ = fused_array(folder, "flamingo")
    working = img_as_float32(flamingo)                              # the step's float32 input (WT1)
    (step,) = [e for e in record["stages"] if e["stage"] == "after_projection"]
    expected = float(np.percentile(working[mask], 5.0, method="inverted_cdf"))
    assert step["fitted"]["low"][0] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    canvas = float(np.percentile(working, 5.0, method="inverted_cdf"))
    assert abs(canvas - expected) > 1e-6 + 1e-6 * abs(expected)    # not the canvas's


# --- AF4: Z > 1 is refused, Z = 1 is allowed ---------------------------------------------------------------------------

@needs_library
@needs_anndata
@pytest.mark.parametrize("recipe, stage", [(InputRecipe(channels=(GaussianConfig(1.0),)), "stage channels"),
                                           (InputRecipe(projection=ProjectionConfig(method="sum")), "projection sum")])
def test_af4_tiles_of_more_than_one_plane_refuse_before_anything_is_written(tmp_path, recipe, stage):
    folder = write_folder(tmp_path, "fused_pair")
    from .test_sample_segmentation_route import export_grid
    grid = export_grid(folder)
    path = nuclei(folder, grid)
    before = directory_files(folder.root)
    with pytest.raises(ValueError, match=stage) as raised:
        folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment="default",
                              config=route_config(segmentation=seeded(path, recipe)), output_dir=folder.export_dir)
    assert "Z = 3" in str(raised.value) and "after_projection" in str(raised.value)
    assert directory_files(folder.root) == before                    # no file changed, no file added


@needs_library
def test_af4_after_projection_steps_pass_and_a_plane_projects_nothing(tmp_path):
    folder, grid = fused(tmp_path / "pair", "fused_pair", 1)
    recipe = InputRecipe(projection=ProjectionConfig(method="max"), after_projection=(GaussianConfig(1.0),))
    record = segment_fused(folder.export_dir, config=seeded(nuclei(folder, grid), recipe), sample=SAMPLE,
                           identity=(DS, SAMPLE)).record
    (projection,) = [e for e in run_entry({"segmentation": record}, "cell")["recipe"]["stages"]
                     if e["stage"] == "projection"]
    assert (projection["applied"], projection["reason"]) == (False, "fused_maximum")
    folder, grid = fused(tmp_path / "plane", "scene_plane", 1)
    record = segment_fused(folder.export_dir, config=seeded(nuclei(folder, grid),
                                                            InputRecipe(projection=ProjectionConfig(method="sum"))),
                           sample=SAMPLE, identity=(DS, SAMPLE)).record
    (projection,) = [e for e in run_entry({"segmentation": record}, "cell")["recipe"]["stages"]
                     if e["stage"] == "projection"]
    assert (projection["applied"], projection["reason"], projection["dtype"]) == (False, "single_plane", "float32")


# --- AF5: check 7 ----------------------------------------------------------------------------------------------------

def check7_config(**run):
    base = {"name": "cell", "target": "cell", "inputs": [{"role": "cytoplasm", "image": "flamingo"}],
            "method": "seeded_watershed", "seeds": "nucleus"}
    return {"backend": "python", "stitching": {"method": "multiview_stitcher", "input": "reference_stain"},
            "assembly": {"assignment": "default"},
            "segmentation": {"input_recipes": {"smooth": {"channels": [{"method": "gaussian", "sigma_um": 1.0}]}}},
            "export": {"route": "after_fusion", "images": ["flamingo=file:images/flamingo/{fov_id}.tif"],
                       "segmentation": {"cells": "cell", "nuclei": "nucleus", "runs": [
                           {"name": "nucleus", "target": "nucleus", "method": "import", "path": "n.tif"},
                           dict(base, **run)]}}}


@pytest.mark.parametrize("run, message", [
    ({"input_recipe": "rough"}, "input_recipe 'rough' is not a recipe of segmentation.input_recipes \\['smooth'\\]"),
    ({"inputs": [{"role": "cytoplasm", "image": "dapi"}]}, "reads the images \\['dapi'\\], which are not names of "
                                                           "export.images"),
    ({"input_recipe": {"combine": {"method": "composite"}}}, "the combine method 'composite' takes the source roles "
                                                              "\\['amplicon', 'nuclear'\\] exactly"),
    ({"method": "stardist", "model": "2D_versatile_fluo", "seeds": None, "target": "nucleus",
      "inputs": [{"role": "nuclear", "image": "flamingo"}]}, "needs input_recipe.model_input")])
def test_af5_check_7_refuses_naming_the_key(run, message):
    config = check7_config(**run)
    if run.get("seeds", "") is None:
        config["export"]["segmentation"]["runs"][1].pop("seeds")
        config["export"]["segmentation"]["cells"] = "nucleus"
    with pytest.raises(ValueError, match=message):
        workflow._check_export_route(config)
    valid = check7_config(input_recipe="smooth")
    workflow._check_export_route(valid)
    export, _ = workflow._export_config(valid)
    assert export.segmentation.runs[1].input_recipe.name == "smooth"


# --- AF6: reuse with another recipe -------------------------------------------------------------------------------------

@needs_library
@needs_anndata
def test_af6_the_reuse_refuses_another_recipe(tmp_path):
    folder = write_folder(tmp_path, "scene_plane")
    from .test_sample_segmentation_route import export_grid
    path = nuclei(folder, export_grid(folder))

    def run_export(recipe, segment):
        return folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment="default",
                                     config=route_config(segmentation=seeded(path, recipe), segment=segment),
                                     output_dir=folder.export_dir).record
    first = run_export(SMOOTH, True)
    recorded = run_entry(first, "cell")["recipe"]["sha256"]
    other = InputRecipe(channels=(GaussianConfig(2.0),))
    from starfinder.segmentation._recipe import recipe_sha256
    before = directory_files(folder.export_dir)
    with pytest.raises(ValueError, match=f"run 'cell' was made with another input recipe: SHA-256 {recorded} "
                                         f"recorded, {recipe_sha256(other)} configured"):
        run_export(other, False)
    assert directory_files(folder.export_dir) == before
    assert run_export(SMOOTH, False)["segmentation"]["status"] == "reused"


# --- AF7: no copy of an unchanged input ---------------------------------------------------------------------------------

@needs_library
def test_af7_the_empty_recipe_writes_no_model_input(tmp_path):
    folder, grid = fused(tmp_path, "scene_plane", 1)
    record = segment_fused(folder.export_dir, config=seeded(nuclei(folder, grid), None), sample=SAMPLE,
                           identity=(DS, SAMPLE)).record
    run = run_entry({"segmentation": record}, "cell")
    _, entry = fused_array(folder, "flamingo")
    assert run["input"]["written"] is False and run["input"]["path"] == entry["path"]
    assert run["input"]["sha256"] == entry["sha256"]
    assert not (folder.export_dir / "model_inputs").exists()
    assert not [o for o in record["outputs"] if o.get("kind") == "model_input"]


# --- AF8 and RE3: the model input files ---------------------------------------------------------------------------------

def collision_config(path, cell_projection, cell):
    """AF8: the images model_input_cell and cell_projection (both the flamingo stain), and two cell runs."""
    extra = (SampleSegmentationRun("cell_projection", "cell", (("cytoplasm", "cell_projection"),),
                                   SeededWatershedConfig(sigma_um=1.0), seeds="nucleus", input_recipe=cell_projection),)
    return seeded(path, cell, image="model_input_cell", extra=extra)


def collision_export(folder, config):
    from dataclasses import replace
    pattern = "file:images/flamingo/{fov_id}.tif"
    export = route_config(segmentation=config)
    export = replace(export, fusion=replace(export.fusion, images=(f"model_input_cell={pattern}",
                                                                   f"cell_projection={pattern}")))
    return folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment="default", config=export,
                                 output_dir=folder.export_dir).record


@needs_library
@needs_anndata
def test_af8_a_model_input_cannot_collide_with_a_fused_image(tmp_path):
    folder = write_folder(tmp_path, "scene_plane")
    from .test_sample_segmentation_route import export_grid
    path = nuclei(folder, export_grid(folder))
    smooth = InputRecipe(channels=(GaussianConfig(1.0),))
    names = ("model_input_cell_projection.ome.tif", "model_inputs/cell_projection.ome.tif", "model_inputs/cell.ome.tif")
    for attempt in (1, 2):
        record = collision_export(folder, collision_config(path, smooth, smooth))
        files = directory_files(folder.export_dir)
        assert set(names) <= set(files) and len({files[n] for n in names}) == 3, attempt
        fused = next(o for o in record["fusion"]["outputs"] if o["path"] == names[0])
        assert files[names[0]] == fused["sha256"]
        for run in ("cell_projection", "cell"):
            assert files[f"model_inputs/{run}.ome.tif"] == run_entry(record, run)["input"]["sha256"]
        check_export(folder.export_dir)                               # R2 (no unlisted file) and T1
    record = collision_export(folder, collision_config(path, None, smooth))
    assert not (folder.export_dir / names[1]).exists() and (folder.export_dir / names[2]).is_file()
    check_export(folder.export_dir)
    collision_export(folder, collision_config(path, None, None))
    assert not (folder.export_dir / "model_inputs").exists()
    check_export(folder.export_dir)


@needs_library
@needs_anndata
def test_re3_the_export_record_and_check_export_cover_the_model_inputs(tmp_path):
    folder = write_folder(tmp_path, "scene_plane")
    from .test_sample_segmentation_route import export_grid
    path = nuclei(folder, export_grid(folder))
    record = folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment="default",
                                   config=route_config(segmentation=seeded(path, SMOOTH)),
                                   output_dir=folder.export_dir).record
    run = run_entry(record, "cell")
    assert {"recipe", "input", "lengths"} <= set(run)
    assert set(run["input"]) == {"path", "sha256", "dtype", "shape_zyxc", "written"}
    assert run["lengths"] == {"sigma_um": {"given": 1.0, "used": 1.0, "rule": "unchanged"}}
    assert run["recipe"]["model_input"] is None and run["recipe"]["config"]["channels"][0]["method"] == "gaussian"
    check_export(folder.export_dir)
    model_input = folder.export_dir / "model_inputs" / "cell.ome.tif"
    original = model_input.read_bytes()
    changed = bytearray(original)
    changed[-1] ^= 0x01
    model_input.write_bytes(bytes(changed))
    with pytest.raises(ValueError, match="check_export (R2|T1)") as raised:
        check_export(folder.export_dir)
    assert "model_inputs/cell.ome.tif" in str(raised.value)
    model_input.write_bytes(original)
    stray = folder.export_dir / "model_inputs" / "stray.ome.tif"
    stray.write_bytes(b"not listed")
    with pytest.raises(ValueError, match="check_export R2") as raised:
        check_export(folder.export_dir)
    assert "model_inputs/stray.ome.tif" in str(raised.value)
    stray.unlink()
    # A listed model input whose run the section does not hold.
    from starfinder.export._check import _t1_model_inputs
    section = copy.deepcopy(record["segmentation"])
    section["runs"] = [r for r in section["runs"] if r["name"] != "cell"]
    with pytest.raises(ValueError, match="check_export T1") as raised:
        _t1_model_inputs(folder.export_dir, section, section["inputs"]["grid"])
    assert "names the run 'cell', which is not a run of the segmentation section" in str(raised.value)


@pytest.mark.parametrize("step", [RecipeStep(GaussianConfig(1.0))])
def test_the_z_rule_names_the_recipe(step):
    from starfinder.export._segment import check_recipe_planes
    config = seeded("n.tif", InputRecipe(channels=(step,), name="smooth"))
    check_recipe_planes(config, 1)
    check_recipe_planes(config, lambda: 1)
    with pytest.raises(ValueError, match="input recipe 'smooth' has steps in stage channels, but the tiles have Z = 4"):
        check_recipe_planes(config, lambda: 4)
