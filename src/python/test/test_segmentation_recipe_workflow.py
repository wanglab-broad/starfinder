"""The input recipe in the workflow configuration: rows MG2, MG5 and WF1 (W-891).

design/validation/segmentation-input-recipe.md: the replaced keys of the YAML are refused naming the recipe field
that replaces them (MG2), the schema's recipe keys equal the configs' fields and its steps the admitted subset of
``PREPROCESSING_METHODS`` (MG5), and ``segmentation.input_recipes`` without runs plans no segment task while runs
with recipes plan the tasks they planned before (WF1). WF1 plans on W-642's fixture (``workflow_examples``).
"""
import copy
import shutil
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from starfinder.dataset import workflow
from starfinder.preprocessing import ProjectionConfig
from starfinder.segmentation import COMBINE_METHODS, InputRecipe, ModelInputConfig
from starfinder.segmentation._recipe import ADMITTED_STEPS, admitted_steps, recipe_fields

from . import workflow_examples as X

pytestmark = [pytest.mark.segmentation, pytest.mark.workflow, pytest.mark.validation, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
NUCLEUS = {"name": "nucleus", "target": "nucleus", "method": "stardist", "model": "2D_versatile_fluo",
           "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "DAPI"}]}


def block(*runs, recipes=None):
    values = {"runs": list(runs)}
    if recipes is not None:
        values["input_recipes"] = recipes
    return {"backend": "python", "segmentation": values}


# --- MG2: the replaced keys of the YAML -------------------------------------------------------------------------

@pytest.mark.parametrize("run, key, field", [
    (dict(NUCLEUS, inputs=[{"role": "composite", "round": "r", "channel": 0,
                            "prepare": {"function": "composite_nuclei_amplicon"}}]), "prepare", "input_recipe.combine"),
    (dict(NUCLEUS, inputs=[{"role": "nuclear", "round": "r", "channel": 0, "prepare_channel": 1}]), "prepare_channel",
     "input_recipe.combine flamingo_enhance"),
    (dict(NUCLEUS, projection={"method": "max"}), "projection", "input_recipe.projection"),
    (dict(NUCLEUS, scale=0.25), "scale", "input_recipe.model_input"),
    (dict(NUCLEUS, method="cellpose", model="cpsam_v2", diameter=30), "diameter", "input_recipe.model_input")])
def test_mg2_the_replaced_keys_of_a_segmentation_run_are_refused(run, key, field):
    with pytest.raises(ValueError) as raised:
        workflow._segmentation_block(block(run))
    message = str(raised.value)
    assert f"{key} is replaced by {field}" in message and "docs/migration.md" in message


@pytest.mark.parametrize("key, value", [("scale", 0.25), ("diameter", 30)])
def test_mg2_the_replaced_keys_of_an_export_run_are_refused(key, value):
    method = "stardist" if key == "scale" else "cellpose"
    config = {"backend": "python", "assembly": {"assignment": "default"},
              "stitching": {"method": "multiview_stitcher", "input": "reference_stain"},
              "export": {"images": ["reference_stain"], "segmentation": {"cells": "cell", "runs": [
                  {"name": "cell", "target": "cell", "method": method, "model_path": "/m",
                   "inputs": [{"role": "nuclear", "image": "reference_stain"}], key: value}]}}}
    with pytest.raises(ValueError, match=f"{key} is replaced by input_recipe.model_input"):
        workflow._export_config(config)


def test_mg2_an_import_keeps_its_own_projection():
    plan, _ = workflow._segmentation_block(block({"name": "nucleus", "target": "nucleus", "method": "import",
                                                  "path": "n.tif", "projection": "max"}))
    assert plan.runs[0].method.projection == "max"
    with pytest.raises(ValueError, match="the import's projection is max, sum or mean"):
        workflow._segmentation_block(block({"name": "nucleus", "target": "nucleus", "method": "import",
                                            "path": "n.tif", "projection": True}))


def test_a_run_gives_a_name_or_a_full_recipe_never_a_name_plus_overrides():
    recipes = {"pi_max": {"projection": {"method": "max"}, "model_input": {"pixel_size_um": 0.37854}}}
    plan, _ = workflow._segmentation_block(block(dict(NUCLEUS, input_recipe="pi_max"), recipes=recipes))
    assert plan.runs[0].input_recipe == InputRecipe(projection=ProjectionConfig(),
                                                    model_input=ModelInputConfig(pixel_size_um=0.37854), name="pi_max")
    with pytest.raises(ValueError, match="unknown .* keys: \\['recipe'\\]"):
        workflow._segmentation_block(block(dict(NUCLEUS, input_recipe={"recipe": "pi_max",
                                                                      "model_input": {"factor": 0.5}}),
                                           recipes=recipes))
    with pytest.raises(ValueError, match="unknown segmentation.input_recipes.pi_max keys: \\['scale'\\]"):
        workflow._segmentation_block(block(dict(NUCLEUS, input_recipe="pi_max"), recipes={"pi_max": {"scale": 1}}))
    with pytest.raises(ValueError, match="empty segmentation block is refused"):
        workflow._segmentation_block({"backend": "python", "segmentation": {"device": "cpu"}})
    plan, device = workflow._segmentation_block({"backend": "python", "segmentation": {"input_recipes": recipes}})
    assert plan is None and device == "cpu"


# --- MG5: the schema keeps the recipe keys equal to the configs -------------------------------------------------------

def test_mg5_the_schema_keys_equal_the_configs():
    defs = SCHEMA["$defs"]
    stages = {f.name for f in fields(InputRecipe) if f.init} - {"name"}
    assert set(defs["input_recipe"]["properties"]) == stages == set(workflow._RECIPE_KEYS)
    steps = {}
    for option in defs["recipe_step"]["oneOf"]:
        properties = defs[option["$ref"].rsplit("/", 1)[1]]["properties"]
        steps[properties["method"]["const"]] = set(properties) - {"method", "roles"}
    assert sorted(steps) == sorted(admitted_steps())
    from starfinder.preprocessing import PREPROCESSING_METHODS
    for config_type in ADMITTED_STEPS:
        assert steps[PREPROCESSING_METHODS[config_type].name] == set(recipe_fields(config_type)), config_type
    combines = {option["properties"]["method"]["const"]: set(option["properties"]) - {"method"}
                for option in defs["recipe_combine"]["oneOf"]}
    assert set(combines) == {spec.name for spec in COMBINE_METHODS.values()}
    for config_type, spec in COMBINE_METHODS.items():
        expected = set() if config_type is type(None) else set(recipe_fields(config_type))
        assert combines[spec.name] == expected, spec.name
    assert set(defs["recipe_projection"]["properties"]) == set(recipe_fields(ProjectionConfig))
    assert set(defs["recipe_model_input"]["properties"]) == {f.name for f in fields(ModelInputConfig)}
    assert "input_recipe" in defs["segmentation_run"]["properties"]
    assert "projection" not in defs["segmentation_run"]["properties"]
    assert not {"prepare", "prepare_channel"} & set(defs["segmentation_input"]["properties"])
    export_run = defs["export_block"]["properties"]["segmentation"]["properties"]["runs"]["items"]
    assert export_run["properties"]["input_recipe"] == {"$ref": "#/$defs/run_input_recipe"}


def test_mg5_the_schema_validates_the_recipe_layout():
    from . import workflow_tasks as T
    config = T.dataset_keys(Path("/work"), morphology={}, segmentation={
        "input_recipes": {"pi_max": {"projection": {"method": "max"}, "model_input": {"pixel_size_um": 0.37854}},
                          "pi_smooth": {"channels": [{"method": "gaussian", "sigma_um": 0.3}],
                                        "projection": {"method": "sum"},
                                        "after_projection": [{"method": "clahe", "kernel_um": 6.0,
                                                              "clip_limit": 0.01}],
                                        "model_input": {"pixel_size_um": 0.37854}}},
        "runs": [dict(NUCLEUS, input_recipe="pi_smooth")]})
    from starfinder.workflow._plan import validate_schema
    validate_schema(config)
    import jsonschema
    validator = jsonschema.Draft202012Validator(dict(SCHEMA["$defs"]["input_recipe"], **{"$defs": SCHEMA["$defs"]}))
    assert not list(validator.iter_errors(config["segmentation"]["input_recipes"]["pi_smooth"]))
    for recipe in ({"channels": [{"method": "white_tophat", "radius_yx": 3}]},
                   {"channels": [{"method": "background_3d", "radius_um_zyx": [1, 1, 1]}]},
                   {"projection": {"method": "max", "conversion": {}}},
                   {"model_input": {"pixel_size_um": 0.4, "factor": 0.5}},
                   {"combine": {"method": "flamingo_enhance", "median_radius_px": 1}}):
        assert list(validator.iter_errors(recipe)), recipe


# --- WF1: planning --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


@pytest.fixture
def work(fixture_work, tmp_path):
    target = tmp_path / "work"
    shutil.copytree(fixture_work, target, symlinks=True)
    return target


def kinds(manifest):
    counts = {}
    for task in manifest["tasks"]:
        counts[task["kind"]] = counts.get(task["kind"], 0) + 1
    return counts


def test_wf1_recipes_without_runs_plan_no_segment_task(work):
    from starfinder.workflow import plan
    config = copy.deepcopy(X.variants(work)["after-fusion"])
    config["segmentation"] = {"input_recipes": {"smooth": {"after_projection": [{"method": "gaussian",
                                                                                   "sigma_um": 1.0}]}}}
    export_runs = config["export"]["segmentation"]["runs"]
    config["export"]["segmentation"].update(nuclei="nucleus", runs=[
        dict(export_runs[0], name="nucleus", target="nucleus"),
        {"name": "cell", "target": "cell", "method": "seeded_watershed", "seeds": "nucleus",
         "inputs": [{"role": "amplicon", "image": "reference_merged"}], "input_recipe": "smooth"}])
    manifest = plan(X.write_config(work, "recipes-only", config))
    assert "segment" not in kinds(manifest) and kinds(manifest) == X.KINDS["after-fusion"]
    from starfinder.workflow._plan import resolve
    resolved = resolve(config, {"path": "<test>", "sha256": None})["configuration"]
    from starfinder.workflow import _translate
    export, _ = _translate.export(resolved)                         # check 7 passed in the plan's resolution
    assert export.segmentation.runs[1].input_recipe.name == "smooth"
    assert resolved["api"]["export"]["route"] == "after_fusion"


def test_wf1_runs_with_recipes_plan_the_tasks_they_planned_before(work):
    from starfinder.workflow import plan
    from starfinder.workflow._plan import _reference_inputs, resolve
    config = copy.deepcopy(X.variants(work)["per-fov"])
    config["segmentation"]["input_recipes"] = {"smooth": {"channels": [{"method": "gaussian", "sigma_um": 1.0}]}}
    manifest = plan(X.write_config(work, "per-fov-recipes", config))
    assert kinds(manifest) == X.KINDS["per-fov"]
    reference_only = X.variants(work)["reference-only"]
    assert reference_only["segmentation"]["runs"][1]["inputs"][1] == {"role": "amplicon", "reference_merged": True}
    resolved = resolve(reference_only, {"path": "<test>", "sha256": None})["configuration"]
    assert _reference_inputs(resolved) == ["cell"]
    assert kinds(plan(X.write_config(work, "reference-only", reference_only))) == X.KINDS["reference-only"]
