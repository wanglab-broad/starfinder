"""RS2, RS3, RS4, SG1, SG2, SG3 and SG6 of docs/sample-segmentation-validation.md: the route setting, check 7
of ``Dataset.export``, the conversion of pixel lengths (LS2) and ``segment_fused`` called alone on a folder in
which ``fuse`` ran; and the route per FOV of ``Dataset.export``, untouched but for the part ``segment``.

Fixtures (``sample_segmentation_fixtures.py``): ``fused_pair`` at ``k`` 1 and 2 and ``fused_quad`` written as
datasets, ``truth_import``, ``seeded_runs``, ``length_configs`` and ``route_configs``; for the route per FOV,
``export_folder`` (``assembly_golden`` as a dataset, ``fusion_fixtures.py``). No model is loaded or run: a
learned method enters through its settings and the model files the test writes. dask runs with the
synchronous scheduler.
"""
import dataclasses
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import tifffile

import starfinder.sample_output._segment as segment_module
from starfinder.assembly import load_assembly
from starfinder.dataset import workflow
from starfinder.image import IncompatibleGeometryError
from starfinder.io.tiff import METADATA_NAMESPACE
from starfinder.sample_output import (ROUTES, ExportConfig, SampleSegmentation, SampleSegmentationConfig,
                                      SampleSegmentationRun, check_export, load_sample_segmentation, segment_fused)
from starfinder.sample_output._check import TOLERANCE_TIFF_UM, _check_tiff
from starfinder.sample_output._segment import LENGTH_FIELDS, check_route, scale_lengths
from starfinder.sample_output._tiff import read_ome_layout
from starfinder.segmentation import (SEGMENTATION_METHODS, CellposeConfig, LabelImportConfig, MissingModelError,
                                     ModelHashMismatchError, SeededWatershedConfig, SegmentationResult,
                                     StarDistConfig)
from starfinder.segmentation._persist import file_sha256

from .assembly_fixtures import SAMPLE as GOLDEN_SAMPLE
from .fusion_fixtures import export_folder, listing
from .sample_segmentation_fixtures import (BIG, DS, SAMPLE, SCENES, fuse_scene, length_configs, model_files,
                                           route_config, seeded_runs, segmentation_grid, truth_import, truth_labels,
                                           truth_runs, write_scene)

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
IDENTITY = (DS, SAMPLE)


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def namespace(run):
    return json.dumps([DS, SAMPLE, None, None, run], separators=(",", ":"))


def record_of(folder):
    return json.loads((folder / "export.json").read_text())


def fused(tmp_path, scene="fused_pair", k=1):
    """The scene written as a dataset and fused alone at ``1/k``; the folder and the segmentation grid."""
    folder = write_scene(tmp_path, SCENES[scene]())
    fuse_scene(folder, k)
    return folder, segmentation_grid(folder.export_dir)


def molecules_only_export(folder, **changes):
    """A route-per-FOV export without cells and without fusion, so that the folder holds files to keep."""
    return folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment=None,
                                 config=ExportConfig(route="per_fov", fuse=False, **changes),
                                 output_dir=folder.export_dir)


def export(folder, config, assignment="default"):
    return folder.dataset.export(SAMPLE, placement=folder.placement_dir, assignment=assignment, config=config,
                                 output_dir=folder.export_dir)


def valid_segmentation(tmp_path):
    """The runs of check 7's valid configuration: a nucleus import, and Cellpose on flamingo with dapi (a model
    file written by the test)."""
    cellpose, _ = model_files(tmp_path)
    nuclei = tmp_path / "truth" / "nucleus.ome.tif"
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei), "nucleus")),
            SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"), ("nuclear", "dapi")),
                                  CellposeConfig(diameter=24.0, model_path=str(cellpose))))
    return SampleSegmentationConfig(runs, "cell", "nucleus")


# --- RS2: a stack and a sum are refused ------------------------------------------------------------------------

RS2_CASES = {"after_fusion_with_stack": dict(route="after_fusion", dimensionality="stack"),
             "after_fusion_with_sum": dict(route="after_fusion", projection="sum"),
             "default_with_sum": dict(route=None, dimensionality="projection", projection="sum")}


@needs_anndata
@pytest.mark.parametrize("assignment", [None, "default"])
@pytest.mark.parametrize("case", list(RS2_CASES))
def test_rs2_a_stack_and_a_sum_are_refused(tmp_path, case, assignment):
    folder = write_scene(tmp_path, SCENES["fused_pair"]())
    molecules_only_export(folder)
    before = listing(folder.root)
    with pytest.raises(ValueError) as caught:
        export(folder, route_config(**RS2_CASES[case]), assignment=assignment)
    message = str(caught.value)
    if case == "after_fusion_with_stack":
        assert "dimensionality 'stack'" in message and "route: per_fov" in message
    else:
        assert "projection" in message and "route" in message and "'sum'" in message
    assert listing(folder.root) == before


# --- RS3: the default without settings (RD1) --------------------------------------------------------------------

@needs_anndata
def test_rs3_the_default_without_settings(tmp_path):
    folder = write_scene(tmp_path, SCENES["fused_pair"]())
    molecules_only_export(folder)
    before = listing(folder.root)
    with pytest.raises(ValueError, match=r"export\.segmentation") as caught:
        export(folder, route_config(route=None, dimensionality="projection"))
    assert "route: per_fov" in str(caught.value)
    assert listing(folder.root) == before
    # The default on the existing export fixture (assembly_golden as a dataset, 3D FOVs, a projection).
    golden = export_folder(tmp_path / "golden")
    with pytest.raises(ValueError, match=r"export\.segmentation"):
        golden.dataset.export(GOLDEN_SAMPLE, placement=golden.placement_dir, output_dir=golden.export_dir)
    assert not golden.export_dir.exists()


# --- RS4: check 7 --------------------------------------------------------------------------------------------------

def _replace_runs(segmentation, runs, **fields):
    """A config whose runs bypass construction (frozen fields set in place), so that check 7 itself meets them."""
    bad = dataclasses.replace(segmentation)
    object.__setattr__(bad, "runs", runs)
    for name, value in fields.items():
        object.__setattr__(bad, name, value)
    return bad


def _rs4_config(tmp_path, violation):
    segmentation = valid_segmentation(tmp_path)
    nucleus, cell = segmentation.runs
    if violation == "image_not_in_images":
        cell = dataclasses.replace(cell, inputs=(("cytoplasm", "membrane_stain"), ("nuclear", "dapi")))
        return route_config(segmentation=dataclasses.replace(segmentation, runs=(nucleus, cell))), ValueError, \
            r"membrane_stain.*export\.images"
    if violation == "seed_naming_a_later_run":
        seeded = SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"),), SeededWatershedConfig(),
                                       seeds="nucleus")
        return route_config(segmentation=_replace_runs(segmentation, (seeded, nucleus))), ValueError, r"seeds"
    if violation == "nuclei_naming_a_cell_run":
        return route_config(segmentation=_replace_runs(segmentation, segmentation.runs, nuclei="cell")), \
            ValueError, r"nuclei"
    if violation == "cells_naming_no_run":
        return route_config(segmentation=_replace_runs(segmentation, segmentation.runs, cells="cytoplasm")), \
            ValueError, r"cells"
    if violation == "fuse_off_with_segment_on":
        return route_config(segmentation=segmentation, fuse=False), ValueError, r"fuse: false with segment: true"
    if violation == "assemble_off_with_segment_on":
        return route_config(segmentation=segmentation, assemble=False), ValueError, \
            r"assemble: false with segment: true"
    if violation == "missing_model_path":
        method = dataclasses.replace(cell.method, model_path=str(tmp_path / "models" / "absent"))
        runs = (nucleus, dataclasses.replace(cell, method=method))
        return route_config(segmentation=dataclasses.replace(segmentation, runs=runs)), MissingModelError, \
            r"run 'cell' model_path"
    if violation == "model_sha256_differs":
        method = dataclasses.replace(cell.method, model_sha256={"cpsam_test": "0" * 64})
        runs = (nucleus, dataclasses.replace(cell, method=method))
        return route_config(segmentation=dataclasses.replace(segmentation, runs=runs)), ModelHashMismatchError, \
            r"run 'cell' model_path"
    raise AssertionError(violation)


RS4_VIOLATIONS = ["image_not_in_images", "seed_naming_a_later_run", "nuclei_naming_a_cell_run", "cells_naming_no_run",
                  "fuse_off_with_segment_on", "assemble_off_with_segment_on", "missing_model_path",
                  "model_sha256_differs"]


@needs_anndata
@pytest.mark.parametrize("violation", RS4_VIOLATIONS)
def test_rs4_check_7(tmp_path, violation):
    folder = write_scene(tmp_path, SCENES["fused_pair"]())
    molecules_only_export(folder)
    before = listing(folder.root)
    config, error, pattern = _rs4_config(tmp_path, violation)
    with pytest.raises(error, match=pattern):
        export(folder, config)
    assert listing(folder.root) == before
    with pytest.raises(error, match=pattern):
        check_route(config, "default")


def test_rs4_check_7_refuses_at_construction_what_it_can():
    """The run and config checks also refuse at construction, before any export."""
    nucleus = SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig("n.tif", "nucleus"))
    cell = SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"),), SeededWatershedConfig(),
                                 seeds="nucleus")
    with pytest.raises(ValueError, match="seeds 'nucleus' is not an earlier run"):
        SampleSegmentationConfig((cell, nucleus), "cell")
    with pytest.raises(ValueError, match="nuclei"):
        SampleSegmentationConfig((nucleus, cell), "cell", nuclei="cell")
    with pytest.raises(ValueError, match="cells"):
        SampleSegmentationConfig((nucleus, cell), "cytoplasm")
    with pytest.raises(ValueError, match="needs seeds"):
        SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"),), SeededWatershedConfig())
    with pytest.raises(ValueError, match="takes no inputs"):
        SampleSegmentationRun("nucleus", "nucleus", (("nuclear", "dapi"),), LabelImportConfig("n.tif", "nucleus"))
    with pytest.raises(TypeError, match="no segmentation method is registered"):
        SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"),), object())
    with pytest.raises(ValueError, match="route"):
        ExportConfig(route="per_sample")
    with pytest.raises(ValueError, match="cuda"):
        SampleSegmentationConfig((nucleus, cell), "cell", device="cuda")
    assert ROUTES == ("per_fov", "after_fusion")


def test_rs4_check_7_unknown_method_key_in_the_workflow_block():
    """The export block's segmentation: a run key that is no field of its method's config is refused, naming it."""
    config = {"backend": "python", "stitching": {"method": "multiview_stitcher", "input": "reference_stain"},
              "assembly": {"assignment": "default"},
              "export": {"images": ["dapi=file:images/dapi/{fov_id}.tif"],
                         "segmentation": {"cells": "cell", "runs": [
                             {"name": "cell", "target": "cell", "inputs": [{"role": "cytoplasm", "image": "dapi"}],
                              "method": "cellpose", "diameter": 24, "model_path": "m", "diamter": 30}]}}}
    with pytest.raises(ValueError, match="diamter"):
        workflow._export_config(config)
    del config["export"]["segmentation"]["runs"][0]["diamter"]
    export_config, _ = workflow._export_config(config)
    run = export_config.segmentation.runs[0]
    assert run.inputs == (("cytoplasm", "dapi"),) and run.method == CellposeConfig(diameter=24, model_path="m")
    assert export_config.route is None and export_config.segment is True


ORDER_CASES = {
    # (violations, the error check 7 raises first)
    "stack_before_segmentation": (dict(route="after_fusion", dimensionality="stack"), "dimensionality 'stack'"),
    "sum_before_segmentation": (dict(projection="sum"), "'sum'"),
    "segmentation_before_images": (dict(segmentation=None, fuse=False), r"export\.segmentation"),
    "images_before_fuse": (dict(image="membrane_stain", fuse=False), r"export\.images"),
    "fuse_before_assemble": (dict(fuse=False, assemble=False), "fuse: false"),
    "assemble_before_models": (dict(assemble=False, model="absent"), "assemble: false"),
}


@pytest.mark.parametrize("case", list(ORDER_CASES))
def test_rs4_check_7_order(tmp_path, case):
    changes, pattern = ORDER_CASES[case]
    changes = dict(changes)
    segmentation = valid_segmentation(tmp_path)
    nucleus, cell = segmentation.runs
    if "image" in changes:
        cell = dataclasses.replace(cell, inputs=(("cytoplasm", changes.pop("image")), ("nuclear", "dapi")))
    if "model" in changes:
        cell = dataclasses.replace(cell, method=dataclasses.replace(
            cell.method, model_path=str(tmp_path / "models" / changes.pop("model"))))
    segmentation = changes.pop("segmentation", dataclasses.replace(segmentation, runs=(nucleus, cell)))
    with pytest.raises(ValueError, match=pattern):
        check_route(route_config(segmentation=segmentation, **changes), "default")


@needs_anndata
def test_rs4_a_valid_configuration_passes_check_7(tmp_path):
    folder = write_scene(tmp_path, SCENES["fused_pair"]())
    molecules_only_export(folder)
    before = listing(folder.root)
    config = route_config(segmentation=valid_segmentation(tmp_path))
    assert check_route(config, "default") == ("after_fusion", "default")
    assert check_route(dataclasses.replace(config, route="after_fusion"), "default") == ("after_fusion", "set")
    assert check_route(route_config(dimensionality="both"), "default") == ("per_fov", "default")
    assert check_route(route_config(dimensionality="stack"), "default") == ("per_fov", "default")
    # Until the route after fusion is part of Dataset.export, a valid configuration is refused after check 7,
    # before anything is written, naming the part that is not available yet.
    with pytest.raises(NotImplementedError, match="not available yet"):
        export(folder, config)
    assert listing(folder.root) == before


# --- The route per FOV is untouched ----------------------------------------------------------------------------

PER_FOV_CASES = {"projection_route_per_fov": dict(route="per_fov"),
                 "stack_without_the_setting": dict(dimensionality="stack"),
                 "stack_route_per_fov": dict(route="per_fov", dimensionality="stack")}


@needs_anndata
@needs_library
@pytest.mark.parametrize("case", list(PER_FOV_CASES))
def test_the_route_per_fov_is_untouched(tmp_path, case):
    """On the existing export fixture: the tables are those assemble gives, the part segment is off with the reason
    route_per_fov, and nothing else of the record names the route."""
    from starfinder.assembly import assemble
    from starfinder.sample_output import FusionConfig
    from starfinder.sample_output._export import assembly_inputs
    changes = dict(PER_FOV_CASES[case])
    dimensionality = changes.pop("dimensionality", "projection")
    folder = export_folder(tmp_path)
    config = ExportConfig(fusion=FusionConfig(dimensionality=dimensionality), **changes)
    record = folder.dataset.export(GOLDEN_SAMPLE, placement=folder.placement_dir, config=config,
                                   output_dir=folder.export_dir).record
    source = "set" if "route" in changes else "default"
    assert record["parts"] == {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["segmentation"] == {"status": "off", "reason": "route_per_fov", "route": "per_fov",
                                      "route_source": source}
    assert not any(name.startswith("segmentation_") for name in listing(folder.export_dir))
    expected = assemble(folder.placement, assembly_inputs(folder.dataset, folder.placement, "default"))
    loaded = load_assembly(folder.export_dir)
    for table in ("fovs", "cells", "molecules", "counts"):
        pd.testing.assert_frame_equal(getattr(loaded, table), getattr(expected, table), check_exact=True)
    assert "route" not in record["assembly"] and "route" not in record["fusion"]
    check_export(folder.export_dir)
    # Without an assignment the part is off with the reason no_cells, on the route per FOV too.
    record = folder.dataset.export(GOLDEN_SAMPLE, placement=folder.placement_dir, assignment=None,
                                   config=dataclasses.replace(config, fuse=False), output_dir=folder.export_dir).record
    assert record["segmentation"] == {"status": "off", "reason": "no_cells", "route": "per_fov",
                                      "route_source": source}


# --- SG2: lengths in pixels (LS2) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_sg2_lengths_in_pixels(tmp_path, k):
    configs = length_configs(tmp_path)
    used, lengths = scale_lengths(configs["cellpose"], k)
    assert used.diameter == 240 / k and used.min_size == max(1, round(15 / k ** 2))
    assert lengths["diameter"] == {"given": 240, "used": 240 / k, "rule": "length"}
    assert lengths["min_size"] == {"given": 15, "used": max(1, round(15 / k ** 2)), "rule": "area"}
    assert {f.name: getattr(used, f.name) for f in dataclasses.fields(used) if f.name not in ("diameter", "min_size")} \
        == {f.name: getattr(configs["cellpose"], f.name) for f in dataclasses.fields(used)
            if f.name not in ("diameter", "min_size")}
    used, lengths = scale_lengths(configs["stardist"], k)
    assert used.scale == 0.25 * k and isinstance(used, StarDistConfig)
    assert lengths["scale"] == {"given": 0.25, "used": 0.25 * k, "rule": "factor_yx"}
    assert lengths["n_tiles"] == {"given": None, "used": None, "rule": "unchanged"}
    used, _ = scale_lengths(dataclasses.replace(configs["stardist"], scale=(1.0, 0.5, 0.5), n_tiles=(2, 2)), k)
    assert used.scale == (1.0, 0.5 * k, 0.5 * k) and used.n_tiles == (2, 2)
    used, lengths = scale_lengths(configs["seeded_watershed"], k)
    assert used == configs["seeded_watershed"] and lengths == {"sigma_um": {"given": 1.5, "used": 1.5,
                                                                           "rule": "unchanged"}}
    blocks = configs["stardist_blocks"]
    if k == 1:
        used, lengths = scale_lengths(blocks, k)
        assert used == blocks
        assert lengths["block_size"] == {"given": [1, 256, 256], "used": [1, 256, 256], "rule": "length_yx"}
    else:
        # The used scale k is not 1, which StarDistConfig refuses together with block fields.
        with pytest.raises(ValueError, match="block fields together with a scale other than 1"):
            scale_lengths(blocks, k)
    # The conversion of the block lengths themselves: ceil(value / k) in Y and X, Z unchanged.
    assert segment_module._scaled((1, 256, 256), "length_yx", k) == (1, -(-256 // k), -(-256 // k))
    assert segment_module._scaled((0, 52, 52), "length_yx", k) == (0, -(-52 // k), -(-52 // k))
    assert segment_module._scaled(52, "length_yx", k) == -(-52 // k)
    imported = LabelImportConfig("cells.tif", "cell")
    assert scale_lengths(imported, k) == (imported, {})


def test_sg2_the_table_of_pixel_fields_equals_the_registered_methods():
    """Every registered method has its row, each listed field is a field of its config, and every other init field
    is one named here as not a length, so a new field of a method config fails until it is classified."""
    not_lengths = {"cellpose": {"model", "model_path", "model_sha256", "do_3d", "anisotropy", "flow_threshold",
                                "cellprob_threshold", "tile_overlap", "bfloat16", "normalize_percentiles"},
                   "stardist": {"model", "model_path", "model_sha256", "prob_thresh", "nms_thresh",
                                "normalize_percentiles"},
                   "seeded_watershed": {"threshold", "compactness", "connectivity"}}
    registered = {spec.name: config for config, spec in SEGMENTATION_METHODS.items()}
    assert set(LENGTH_FIELDS) == set(registered) == set(not_lengths)
    for name, config in registered.items():
        init = {f.name for f in dataclasses.fields(config) if f.init}
        assert set(LENGTH_FIELDS[name]) <= init
        assert set(LENGTH_FIELDS[name]) | not_lengths[name] == init and not set(LENGTH_FIELDS[name]) & not_lengths[name]


# --- SG1: the input is the fused projection ---------------------------------------------------------------------

def _capture(monkeypatch, stub=None):
    captured = []
    real = segment_module.segment

    def wrapped(segmentation_input, **kwargs):
        captured.append((segmentation_input, kwargs))
        return (stub or real)(segmentation_input, **kwargs)
    monkeypatch.setattr(segment_module, "segment", wrapped)
    return captured


def _fused_projection(folder, image):
    entry = next(o for o in record_of(folder.export_dir)["fusion"]["outputs"]
                 if o["image"] == image and o["format"] == "ome_tiff")
    return tifffile.imread(folder.export_dir / entry["path"]), entry


def _assert_grid_is_the_fusion_output(grid, folder):
    for image in ("dapi", "flamingo"):
        _, entry = _fused_projection(folder, image)
        assert list(grid.shape_zyx) == entry["shape_zyx"]
        assert list(grid.metadata.spacing_zyx) == entry["spacing_zyx"]
        assert list(grid.metadata.origin_zyx) == entry["origin_zyx"]
    assert grid.metadata.frame_id == f"sample:{SAMPLE}" and grid.metadata.spatial_unit == "µm"


@needs_library
@pytest.mark.parametrize("scene,k", [("fused_pair", 1), ("fused_pair", 2), ("fused_quad", 1)])
def test_sg1_the_input_is_the_fused_projection(tmp_path, monkeypatch, scene, k):
    folder, grid = fused(tmp_path, scene, k)
    config = seeded_runs(truth_import(folder, grid, "nucleus"))
    captured = _capture(monkeypatch)
    result = segment_fused(folder.export_dir, config=config, sample=SAMPLE, identity=IDENTITY)
    assert len(captured) == 1
    segmentation_input, kwargs = captured[0]
    flamingo, _ = _fused_projection(folder, "flamingo")
    np.testing.assert_array_equal(segmentation_input.image, flamingo[np.newaxis, ..., np.newaxis])
    assert segmentation_input.image.dtype == flamingo.dtype and segmentation_input.roles == ("cytoplasm",)
    _assert_grid_is_the_fusion_output(segmentation_input.grid, folder)
    assert segmentation_input.grid.metadata == grid.metadata and segmentation_input.grid.shape_zyx == grid.shape_zyx
    assert kwargs["label_namespace"] == namespace("cell") and kwargs["target"] == "cell"
    assert kwargs["seeds"] is result.runs["nucleus"] and kwargs["device"] == "cpu"
    assert kwargs["config"] == config.runs[1].method
    for name, run in result.runs.items():
        assert run.label_namespace == namespace(name) and run.grid == segmentation_input.grid
        assert run.geometry == "plane"
    assert segmentation_input.sources[0]["image"] == "flamingo"
    assert segmentation_input.sources[0]["sha256"] == file_sha256(folder.export_dir / "flamingo_projection.ome.tif")


@needs_library
def test_sg1_channels_in_the_run_s_role_order(tmp_path, monkeypatch):
    """A run of two roles: the input's channels are the fused projections in the run's order. The call to the method
    is replaced by a stub that returns the nucleus labels, so no model is loaded."""
    folder, grid = fused(tmp_path, "fused_pair", 2)
    nuclei = truth_import(folder, grid, "nucleus")
    cellpose, _ = model_files(tmp_path)
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei), "nucleus")),
            SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"), ("nuclear", "dapi")),
                                  CellposeConfig(diameter=24.0, min_size=15, model_path=str(cellpose))))

    def stub(segmentation_input, *, config, target, seeds, device, label_namespace):
        labels = truth_labels(folder.scene, segmentation_input.grid, "nucleus")
        return SegmentationResult(labels, segmentation_input.grid, target, "plane", label_namespace,
                                  {"methods": [{"method": "stub"}], "seeds": None, "outcome": "ok"})
    captured = _capture(monkeypatch, stub)
    result = segment_fused(folder.export_dir, config=SampleSegmentationConfig(runs, "cell", "nucleus"), sample=SAMPLE,
                           identity=IDENTITY)
    segmentation_input, kwargs = captured[0]
    flamingo, _ = _fused_projection(folder, "flamingo")
    dapi, _ = _fused_projection(folder, "dapi")
    np.testing.assert_array_equal(segmentation_input.image, np.stack([flamingo, dapi], axis=-1)[np.newaxis])
    assert segmentation_input.roles == ("cytoplasm", "nuclear")
    assert [s["image"] for s in segmentation_input.sources] == ["flamingo", "dapi"]
    # The lengths at k = 2 (LS2).
    assert kwargs["config"].diameter == 12.0 and kwargs["config"].min_size == 4
    assert result.record["runs"][1]["lengths"]["diameter"] == {"given": 24.0, "used": 12.0, "rule": "length"}


# --- SG3: label files and the section ----------------------------------------------------------------------------

RECORD_KEYS = {"status", "route", "inputs", "runs", "cells", "nuclei", "device", "outputs", "software", "wall_s",
               "cpu_s", "peak_rss_kib"}
RUN_KEYS = {"name", "target", "geometry", "label_namespace", "inputs", "seeds", "method", "import", "lengths", "path",
            "sha256", "n_labels", "max_label", "outcome"}


def _t1(folder, section, record):
    """T1 of docs/sample-segmentation-contract.md ("The record"), evaluated here."""
    grid = section["inputs"]["grid"]
    for run in section["runs"]:
        path = folder / run["path"]
        assert path.is_file() and file_sha256(path) == run["sha256"]
        values = tifffile.imread(path)
        assert values.dtype == np.uint32 and [1, *values.shape] == grid["shape_zyx"]
        layout = read_ome_layout(path)
        metadata = layout["annotations"][METADATA_NAMESPACE]
        assert metadata["frame_id"] == grid["frame"] and metadata["spatial_unit"] == grid["unit"]
        assert np.allclose(metadata["spacing_zyx"], grid["spacing_zyx"], rtol=0, atol=TOLERANCE_TIFF_UM)
        assert np.allclose(metadata["origin_zyx"], grid["origin_zyx"], rtol=0, atol=TOLERANCE_TIFF_UM)
    if record["fusion"]["status"] == "written":
        outputs = {o["image"]: o for o in record["fusion"]["outputs"] if o["dimensionality"] == "projection"
                   and o["format"] == "ome_tiff"}
        for entry in section["inputs"]["images"]:
            fused_entry = outputs[entry["image"]]
            assert fused_entry["path"] == entry["path"]
            assert file_sha256(folder / fused_entry["path"]) == entry["sha256"] == fused_entry["sha256"]
            assert fused_entry["shape_zyx"] == grid["shape_zyx"]
            assert np.allclose(fused_entry["spacing_zyx"], grid["spacing_zyx"], rtol=0, atol=TOLERANCE_TIFF_UM)
            assert np.allclose(fused_entry["origin_zyx"], grid["origin_zyx"], rtol=0, atol=TOLERANCE_TIFF_UM)


@needs_library
@pytest.mark.parametrize("runs", ["seeded_runs", "truth_import"])
@pytest.mark.parametrize("scene,k", [("fused_pair", 1), ("fused_pair", 2), ("fused_quad", 1)])
def test_sg3_label_files_and_the_section(tmp_path, scene, k, runs):
    folder, grid = fused(tmp_path, scene, k)
    nuclei = truth_import(folder, grid, "nucleus")
    if runs == "seeded_runs":
        config = seeded_runs(nuclei)
    else:
        config = truth_runs(truth_import(folder, grid, "cell"), nuclei)
    result = segment_fused(folder.export_dir, config=config, sample=SAMPLE, identity=IDENTITY)
    assert isinstance(result, SampleSegmentation) and result.status == "written"
    record = record_of(folder.export_dir)
    section = record["segmentation"]
    assert result.record == section and record["parts"]["segment"] == "written"
    assert record["parts"] == {"assemble": "off", "fuse": "written", "segment": "written", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    # One file per run, uint32, the OME-TIFF layout of the fused image, on the recorded grid.
    names = sorted(p for p in listing(folder.export_dir) if p.startswith("segmentation_"))
    assert names == sorted(f"segmentation_{r.name}.ome.tif" for r in config.runs)
    assert [o["path"] for o in section["outputs"]] == [f"segmentation_{r.name}.ome.tif" for r in config.runs]
    for output in section["outputs"]:
        _check_tiff(folder.export_dir, output["path"], output, section)
        assert output["dtype"] == "uint32"
    # Every field of "The record".
    assert RECORD_KEYS <= set(section) and section["route"] == "after_fusion"
    assert section["inputs"]["grid"] == {"shape_zyx": list(grid.shape_zyx),
                                         "spacing_zyx": list(grid.metadata.spacing_zyx),
                                         "origin_zyx": list(grid.metadata.origin_zyx), "frame": f"sample:{SAMPLE}",
                                         "unit": "µm", "factor_yx": k, "projection": "max"}
    assert [e["image"] for e in section["inputs"]["images"]] == (["flamingo"] if runs == "seeded_runs" else [])
    assert section["inputs"]["placement"] == {"fovs_sha256": record["fusion"]["placement"]["fovs_sha256"],
                                              "links_sha256": record["fusion"]["placement"]["links_sha256"]}
    assert not any("fusion" in key for key in json.dumps(section["inputs"]).split('"') if key.endswith("sha256"))
    assert (section["cells"], section["nuclei"], section["device"]) == ("cell", "nucleus", "cpu")
    for run, entry in zip(config.runs, section["runs"]):
        assert set(entry) == RUN_KEYS and entry["name"] == run.name and entry["geometry"] == "plane"
        assert entry["label_namespace"] == namespace(run.name)
        assert entry["inputs"] == [{"role": role, "image": image} for role, image in run.inputs]
        labels = result.runs[run.name].labels
        assert entry["n_labels"] == int(np.count_nonzero(np.unique(labels))) and entry["max_label"] == int(labels.max())
        if isinstance(run.method, LabelImportConfig):
            assert entry["method"] is None and entry["import"]["relabel"] is False and entry["lengths"] == {}
        else:
            assert entry["method"]["method"] == "seeded_watershed" and entry["import"] is None
            assert entry["seeds"]["run"] == "nucleus"
            assert entry["lengths"] == {"sigma_um": {"given": 1.0, "used": 1.0, "rule": "unchanged"}}
    _t1(folder.export_dir, section, record)
    # The truth IDs, and those at and above 2^31 of fused_quad, are kept in the files.
    if runs == "truth_import":
        for target in ("cell", "nucleus"):
            np.testing.assert_array_equal(tifffile.imread(folder.export_dir / f"segmentation_{target}.ome.tif"),
                                          truth_labels(folder.scene, grid, target)[0])
        if scene == "fused_quad":
            assert {BIG, 2 ** 32 - 1} <= set(np.unique(result.runs["cell"].labels).tolist())
            assert section["runs"][1]["max_label"] == 2 ** 32 - 1
    # The folder is equal to its record; the read-back checks of the present conditions pass.
    check_export(folder.export_dir)
    loaded = load_sample_segmentation(folder.export_dir)
    assert loaded.grid == result.grid and list(loaded.runs) == list(result.runs)
    for name, run in loaded.runs.items():
        np.testing.assert_array_equal(run.labels, result.runs[name].labels)
        assert run.label_namespace == result.runs[name].label_namespace and run.target == result.runs[name].target
    assert loaded.cells is loaded.runs["cell"] and loaded.nuclei is loaded.runs["nucleus"]


@needs_library
def test_sg3_load_sample_segmentation_checks_the_label_files(tmp_path):
    folder, grid = fused(tmp_path, "fused_pair", 1)
    with pytest.raises(ValueError, match="holds no saved sample segmentation"):
        load_sample_segmentation(folder.export_dir)
    nuclei = truth_import(folder, grid, "nucleus")
    segment_fused(folder.export_dir, config=seeded_runs(nuclei), sample=SAMPLE, identity=IDENTITY)
    path = folder.export_dir / "segmentation_cell.ome.tif"
    recorded = record_of(folder.export_dir)["segmentation"]["runs"][1]["sha256"]
    labels = tifffile.imread(path)
    labels[0, 0] = 99
    tifffile.imwrite(path, labels)
    with pytest.raises(ValueError, match="SHA-256") as caught:
        load_sample_segmentation(folder.export_dir)
    assert str(path) in str(caught.value) and recorded in str(caught.value) and file_sha256(path) in str(caught.value)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_sample_segmentation(folder.export_dir)


@needs_library
def test_sg3_segment_fused_again_replaces_the_files_of_its_earlier_section(tmp_path):
    folder, grid = fused(tmp_path, "fused_pair", 1)
    nuclei = truth_import(folder, grid, "nucleus")
    segment_fused(folder.export_dir, config=truth_runs(truth_import(folder, grid, "cell"), nuclei), sample=SAMPLE,
                  identity=IDENTITY)
    runs = (SampleSegmentationRun("nuclei_only", "nucleus", (), LabelImportConfig(str(nuclei), "nucleus")),)
    segment_fused(folder.export_dir, config=SampleSegmentationConfig(runs, "nuclei_only", "nuclei_only"),
                  sample=SAMPLE, identity=IDENTITY)
    assert sorted(p for p in listing(folder.export_dir) if p.startswith("segmentation_")) == \
        ["segmentation_nuclei_only.ome.tif"]
    check_export(folder.export_dir)


def test_sg3_segment_fused_without_a_fusion_records_not_run(tmp_path):
    folder = write_scene(tmp_path, SCENES["fused_pair"]())
    nuclei = tmp_path / "nuclei.tif"
    from starfinder.sample_output import ExportIncompleteWarning
    with pytest.warns(ExportIncompleteWarning, match="part_not_run"):
        result = segment_fused(folder.export_dir, config=seeded_runs(nuclei), sample=SAMPLE, identity=IDENTITY)
    assert result.status == "not_run" and result.record["reason"] == "part_not_run" and result.runs == {}
    record = record_of(folder.export_dir)
    assert record["parts"]["segment"] == "not_run" and record["parts"]["fuse"] == "off"
    assert record["complete"] is False and sorted(listing(folder.export_dir)) == ["export.json"]


# --- SG6: import runs -------------------------------------------------------------------------------------------------

@needs_library
def test_sg6_import_runs(tmp_path):
    folder, grid = fused(tmp_path, "fused_pair", 1)
    cells = truth_labels(folder.scene, grid, "cell")
    # Values that are not 1..n: relabel false keeps them.
    cells = np.where(cells > 0, cells * np.uint32(7), 0).astype(np.uint32)
    config = truth_runs(truth_import(folder, grid, "cell", labels=cells), truth_import(folder, grid, "nucleus"))
    result = segment_fused(folder.export_dir, config=config, sample=SAMPLE, identity=IDENTITY)
    assert sorted(np.unique(result.runs["cell"].labels).tolist()) == [0, 7, 14, 21]
    assert result.record["runs"][1]["import"]["relabel"] is False
    np.testing.assert_array_equal(result.runs["cell"].labels, cells)
    # The same file on a grid one pixel larger: refused before anything is written.
    before = listing(folder.export_dir)
    larger = np.zeros((1, grid.shape_zyx[1] + 1, grid.shape_zyx[2]), np.uint32)
    larger[:, :-1] = cells
    bigger = truth_import(folder, grid, "cell", path=tmp_path / "bigger.ome.tif", labels=larger)
    with pytest.raises(IncompatibleGeometryError):
        segment_fused(folder.export_dir, config=truth_runs(bigger, tmp_path / "output" / "truth" / "nucleus.ome.tif"),
                      sample=SAMPLE, identity=IDENTITY)
    assert listing(folder.export_dir) == before
