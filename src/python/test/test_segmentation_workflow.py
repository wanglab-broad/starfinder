"""The §2.9 segmentation workflow key: the legacy stardist_segmentation translation, the Python-only
segmentation block, the static schema, the adapter run and the shared rules under both backends
(W-317; docs/segmentation-contract.md, "Workflow configuration")."""
import copy
import glob
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
import tifffile
import yaml

from starfinder._registry import names
from starfinder.dataset.workflow import _run_stardist_segmentation, _segmentation, _segmentation_block
from starfinder.preprocessing import ProjectionConfig
from starfinder.segmentation import (SEGMENTATION_METHODS, CellposeConfig, CompositeConfig, ExpandLabelsConfig,
                                     InputChannel, LabelImportConfig, SeededWatershedConfig, SegmentationPlan,
                                     SegmentationRun, StarDistConfig)

from .test_segmentation_golden import DISTANCE, LABELS_3D, digest, fixture, stand_in_model

pytestmark = [pytest.mark.segmentation, pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
SHARED_SCRIPTS = ("stardist_segmentation.py", "create_nuclei_amplicon_overlay.py", "enhance_dapi_with_flamingo.py")
STORED = {"prob": 0.5, "nms": 0.4}


def model_folder(path, n_dim=3):
    """A stand-in StarDist model folder: the three files the resolution hashes, with stored thresholds."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"n_dim": n_dim, "grid": [1] * n_dim}))
    (path / "thresholds.json").write_text(json.dumps(STORED))
    (path / "weights_best.h5").write_bytes(b"stand-in weights")
    return path


def workflow(tmp_path, backend="python", **parameters):
    values = {"stardist_base_path": str(tmp_path / "models"), "stardist_model_name": "stand_in",
              "segmentation_input_folder": "DAPI", "prob_thresh": 0.5, "nms_thresh": 0.4, "rescale": False,
              "expand_labels": True, "distance": DISTANCE, **parameters}
    return {"backend": backend, "dataset_id": "data", "sample_id": "sample", "output_id": "out",
            "root_output_path": str(tmp_path / "output"), "voxel_size_z": 0.35, "voxel_size_xy": 0.1,
            "rotate_angle": 90, "dapi_round": "round_that_does_not_exist",
            "rules": {"stardist_segmentation": {"run": True, "parameters": values},
                      "create_nuclei_amplicon_overlay": {"run": True, "parameters": {"maximum_projection": False}}}}


# --- The translation of the legacy keys ----------------------------------------------------------------

def test_the_legacy_keys_translate_to_one_stardist_run(tmp_path):
    model_folder(tmp_path / "models" / "stand_in")
    legacy = _segmentation(workflow(tmp_path))
    assert legacy.config == StarDistConfig(scale=1.0, model_path=str(tmp_path / "models" / "stand_in"))
    assert legacy.operations == (ExpandLabelsConfig(DISTANCE, "pixel", "planar"),)
    assert (legacy.target, legacy.role, legacy.device, legacy.projection) == ("nucleus", "nuclear", "cpu", None)
    assert legacy.record["threshold_source"] == "stored" and legacy.record["stored_thresholds"] == STORED
    assert legacy.record["target_source"] == "legacy_default" and legacy.record["model"] == "path"


def test_thresholds_other_than_the_stored_ones_are_overrides(tmp_path):
    model_folder(tmp_path / "models" / "stand_in")
    legacy = _segmentation(workflow(tmp_path, prob_thresh=0.6))
    assert (legacy.config.prob_thresh, legacy.config.nms_thresh) == (0.6, None)
    assert legacy.record["threshold_source"] == "override"
    # Without the model's thresholds.json nothing is known to be stored: the values are overrides.
    legacy = _segmentation(workflow(tmp_path, stardist_model_name="absent"))
    assert (legacy.config.prob_thresh, legacy.config.nms_thresh) == (0.5, 0.4)
    assert legacy.record["threshold_source"] == "override"


def test_a_known_model_in_the_weights_cache_resolves_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(tmp_path / "weights"))
    model_folder(tmp_path / "weights" / "stardist" / "2D_versatile_fluo", n_dim=2)
    config = workflow(tmp_path, stardist_base_path=str(tmp_path / "weights" / "stardist"),
                      stardist_model_name="2D_versatile_fluo")
    legacy = _segmentation(config)
    assert legacy.config.model == "2D_versatile_fluo" and legacy.config.model_path is None
    assert legacy.record["model"] == "known" and legacy.record["threshold_source"] == "stored"
    # The same name under another base path is a user model given by path.
    config["rules"]["stardist_segmentation"]["parameters"]["stardist_base_path"] = str(tmp_path / "other")
    assert _segmentation(config).config.model_path == str(tmp_path / "other" / "2D_versatile_fluo")


@pytest.mark.parametrize("rescale, scale", [(False, 1.0), (True, 0.5)])
def test_rescale_is_the_stardist_scale(tmp_path, rescale, scale):
    assert _segmentation(workflow(tmp_path, rescale=rescale)).config.scale == scale


def test_no_expansion_without_expand_labels(tmp_path):
    assert _segmentation(workflow(tmp_path, expand_labels=False)).operations == ()
    parameters = workflow(tmp_path)
    del parameters["rules"]["stardist_segmentation"]["parameters"]["distance"]
    with pytest.raises(ValueError, match="needs distance"):
        _segmentation(parameters)


@pytest.mark.parametrize("folder, role, target", [("overlay", "composite", "cell"), ("DAPI", "nuclear", "nucleus"),
                                                  ("flamingo/enhanced_DAPI", "nuclear", "nucleus"),
                                                  ("custom", "nuclear", "nucleus")])
def test_the_input_folder_gives_the_role_and_the_default_target(tmp_path, folder, role, target):
    legacy = _segmentation(workflow(tmp_path, segmentation_input_folder=folder))
    assert (legacy.role, legacy.target, legacy.record["target_source"]) == (role, target, "legacy_default")


def test_the_python_only_target_and_device(tmp_path):
    legacy = _segmentation(workflow(tmp_path, segmentation_input_folder="overlay", target="nucleus", device="cuda"))
    assert (legacy.target, legacy.record["target_source"], legacy.device) == ("nucleus", "config", "cuda")
    for key, value in (("target", "nucleus"), ("device", "cpu")):
        with pytest.raises(ValueError, match=f"parameters.{key} is Python-only"):
            _segmentation(workflow(tmp_path, backend="matlab", **{key: value}))
    with pytest.raises(ValueError, match="nucleus or cell"):
        _segmentation(workflow(tmp_path, target="membrane"))


def test_the_projection_that_made_the_input_is_recorded(tmp_path):
    config = workflow(tmp_path, segmentation_input_folder="overlay")
    assert _segmentation(config).projection is None
    config["rules"]["create_nuclei_amplicon_overlay"]["parameters"]["maximum_projection"] = True
    assert _segmentation(config).projection == ProjectionConfig()
    config = workflow(tmp_path, segmentation_input_folder="DAPI")
    config["rules"]["create_nuclei_amplicon_overlay"]["parameters"]["maximum_projection"] = True
    assert _segmentation(config).projection is None  # the overlay's projection does not reach the DAPI folder
    assert _segmentation(dict(config, maximum_projection=True)).projection == ProjectionConfig()


def test_unknown_keys_and_the_block_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown stardist_segmentation parameter keys"):
        _segmentation(workflow(tmp_path, n_tiles=[1, 4, 4]))
    with pytest.raises(ValueError, match="FOV.segment"):
        _segmentation(dict(workflow(tmp_path), segmentation={"runs": []}))


# --- The adapter run -------------------------------------------------------------------------------------

def stand_in_run(image, config, context):
    """The W-307 stand-in model in place of normalize + predict_instances."""
    return stand_in_model(image[..., 0]), {"effective": {"stand_in": True}}


@pytest.fixture
def stand_in_stardist(monkeypatch):
    """The stardist method with its run replaced by the stand-in, and no backend import (no TensorFlow)."""
    spec = SEGMENTATION_METHODS[StarDistConfig]
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig, replace(spec, run=stand_in_run, requires=()))


def run_segmentation(tmp_path, config, image):
    path = tmp_path / "output" / "data" / "out" / "images" / "DAPI" / "Position001.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, image)
    output = tmp_path / "output" / "data" / "out" / "images" / "stardist_segmentation" / "Position001.tif"
    _run_stardist_segmentation(SimpleNamespace(input=[str(path)], output=[str(output)], config=config,
                                               wildcards=SimpleNamespace(fovID="Position001")))
    return tifffile.imread(output), json.loads(output.with_suffix(".json").read_text())


@pytest.mark.parametrize("backend", ["python", "matlab"])
def test_the_rule_writes_uint32_labels_equal_to_the_legacy_pins(tmp_path, stand_in_stardist, backend):
    """The translated legacy run (rescale false, expand_labels true, distance 4) gives LABELS_3D[(False, True)]."""
    model_folder(tmp_path / "models" / "stand_in")
    labels, record = run_segmentation(tmp_path, workflow(tmp_path, backend=backend), fixture()["dapi"])
    assert labels.dtype == np.uint32 and labels.shape == (16, 64, 64)
    assert digest(labels.astype("uint16")) == LABELS_3D[(False, True)]
    assert record["labels"]["dtype"] == "uint32" and record["target"] == "nucleus"
    assert [op["operation"] for op in record["operations"]] == ["expand_labels"]
    assert record["workflow"]["metadata_source"] == "declared" and record["workflow"]["target_source"] == "legacy_default"
    assert record["grid"]["metadata"]["spacing_zyx"] == [0.35, 0.1, 0.1]
    assert record["methods"][0]["effective"] == {"stand_in": True}


def test_an_image_without_foreground_gives_an_empty_label_image(tmp_path, monkeypatch):
    """No foreground gate: an all-zero image is outcome empty, where the script raised (W-307 golden test)."""
    model_folder(tmp_path / "models" / "stand_in")

    def empty_run(image, config, context):
        return np.zeros(image.shape[:3], np.int32), {}

    spec = SEGMENTATION_METHODS[StarDistConfig]
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig, replace(spec, run=empty_run, requires=()))
    labels, record = run_segmentation(tmp_path, workflow(tmp_path, expand_labels=False),
                                      np.zeros((16, 64, 64), np.uint8))
    assert labels.dtype == np.uint32 and not labels.any() and record["outcome"] == "empty"


# --- The Python-only block ------------------------------------------------------------------------------

BLOCK = yaml.safe_load("""
segmentation:
  device: cpu
  runs:
    - name: nucleus
      target: nucleus
      inputs:
        - {role: nuclear, round: round4, channel: ch04}
      method: stardist
      model_path: /absolute/stardist_models/3D_spleen
      scale: 1.0
      operations:
        - {operation: expand_labels, distance: 4, unit: pixel, mode: planar}
    - name: cell
      target: cell
      seeds: nucleus
      inputs:
        - {role: amplicon, reference_merged: true}
      method: seeded_watershed
      sigma_um: 1.5
""")


def test_the_block_of_the_contract_gives_a_plan():
    plan, device = _segmentation_block({"backend": "python", **BLOCK})
    assert device == "cpu"
    assert plan == SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", "round4", "ch04"),),
                        StarDistConfig(scale=1.0, model_path="/absolute/stardist_models/3D_spleen"),
                        operations=(ExpandLabelsConfig(4, "pixel", "planar"),)),
        SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                        SeededWatershedConfig(sigma_um=1.5), seeds="nucleus")))


def test_block_imports_prepare_and_projection():
    runs = [{"name": "nucleus", "target": "nucleus", "method": "import", "path": "/labels/nuclei.tif",
             "relabel": True},
            {"name": "cell", "target": "cell", "method": "cellpose", "model": "cpsam_v2", "diameter": 30,
             "projection": True,
             "inputs": [{"role": "cytoplasm", "round": "round1", "channel": 0,
                         "prepare": {"function": "composite_nuclei_amplicon", "nuclear_quantile": 0.01}}]}]
    plan, device = _segmentation_block({"backend": "python", "segmentation": {"device": "cuda", "runs": runs}})
    assert device == "cuda"
    assert plan.runs[0].method == LabelImportConfig("/labels/nuclei.tif", "nucleus", relabel=True)
    assert plan.runs[1].method == CellposeConfig(diameter=30, model="cpsam_v2")
    assert plan.runs[1].projection == ProjectionConfig()
    assert plan.runs[1].inputs[0].prepare == CompositeConfig(nuclear_quantile=0.01)


@pytest.mark.parametrize("change, error, match", [
    ({"backend": "matlab"}, ValueError, "Python-only"),
    ({"runs": [{"name": "x", "target": "cell", "method": "watershed"}]}, ValueError, "unknown segmentation method"),
    ({"runs": [{"name": "x", "target": "cell", "method": "cellpose", "model": "cpsam_v2"}]}, ValueError,
     "requires diameter"),
    ({"runs": [{"name": "x", "target": "nucleus", "method": "stardist", "scale": 1.0, "model": "2D_versatile_fluo",
                "tile_size": 4}]}, ValueError, "unknown segmentation run 'x' \\(stardist\\) keys"),
    ({"runs": []}, ValueError, "nonempty list"),
    ({"device": "gpu"}, ValueError, "segmentation.device"),
])
def test_invalid_blocks_raise(change, error, match):
    config = {"backend": "python", "segmentation": copy.deepcopy(BLOCK["segmentation"])}
    config["segmentation"].update({k: v for k, v in change.items() if k != "backend"})
    config.update({k: v for k, v in change.items() if k == "backend"})
    with pytest.raises(error, match=match):
        _segmentation_block(config)


# --- The static schema ----------------------------------------------------------------------------------

def test_schema_segmentation_methods_equal_the_registry():
    methods = SCHEMA["$defs"]["segmentation_run"]["properties"]["method"]["anyOf"]
    assert set(methods[0]["enum"]) == set(names(SEGMENTATION_METHODS))
    assert methods[1] == {"const": "import"}


def base_config():
    config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
    config["backend"] = "python"
    return config


@pytest.mark.parametrize("addition", [
    {"segmentation": BLOCK["segmentation"]},
    {"rules": {"stardist_segmentation": {"parameters": {"target": "cell"}}}},
    {"rules": {"stardist_segmentation": {"parameters": {"device": "cuda"}}}},
])
def test_python_only_keys_validate_on_the_python_backend_only(addition):
    config = base_config()
    for key, value in addition.items():
        if key == "rules":
            config["rules"]["stardist_segmentation"]["parameters"].update(value["stardist_segmentation"]["parameters"])
        else:
            config[key] = value
    jsonschema.validate(config, SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(config, backend="matlab"), SCHEMA)


def test_the_schema_rejects_an_unknown_method_and_target():
    config = dict(base_config(), segmentation={"runs": [{"name": "x", "target": "cell", "method": "watershed"}]})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    config = base_config()
    config["rules"]["stardist_segmentation"]["parameters"]["target"] = "membrane"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)


# --- The shared rules under both backends ---------------------------------------------------------------

def test_the_shared_rules_do_not_branch_on_the_backend_and_read_no_envs_path():
    """Amendment 3 of W-309: both backends run the same adapter calls; envs_path stays in the schema only."""
    texts = [(ROOT / "workflow/scripts" / name).read_text() for name in SHARED_SCRIPTS]
    texts.append((ROOT / "workflow/rules/segmentation.smk").read_text())
    assert not [t for t in texts if re.search("backend", t, re.IGNORECASE)]
    assert not [p for p in (ROOT / "workflow/rules").iterdir() if "envs_path" in p.read_text()]
    assert "envs_path" in SCHEMA["properties"]
    assert "conda:" not in texts[-1]


def get_dapi_input(path, input_dir, dapi_round="round1"):
    """get_dapi_input of a rule file, executed with the names common.smk provides."""
    text = path.read_text()
    source = re.search(r"^def get_dapi_input\(wildcards\):\n(?:    .*\n)+", text, re.MULTILINE).group(0)
    namespace = {"glob": glob, "INPUT_DIR": input_dir, "config": {"dapi_round": dapi_round}}
    exec(source, namespace)
    return namespace["get_dapi_input"]


@pytest.mark.parametrize("rules", ["registration.smk", "registration-py.smk"])
def test_get_dapi_input_needs_exactly_one_file(tmp_path, rules):
    function = get_dapi_input(ROOT / "workflow/rules" / rules, tmp_path)
    folder = tmp_path / "round1" / "Position001"
    folder.mkdir(parents=True)
    wildcards = SimpleNamespace(fovID="Position001")
    with pytest.raises(ValueError, match="found 0"):
        function(wildcards)
    (folder / "a_ch04.tif").write_bytes(b"")
    assert function(wildcards) == [str(folder / "a_ch04.tif")]
    (folder / "b_ch04.tif").write_bytes(b"")
    with pytest.raises(ValueError, match="found 2.*a_ch04.tif.*b_ch04.tif"):
        function(wildcards)
