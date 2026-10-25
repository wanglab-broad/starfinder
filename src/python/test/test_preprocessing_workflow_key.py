"""The Python-only preprocessing workflow key: adapter, static schema and its consistency with PREPROCESSING_METHODS (W-232)."""
import copy
from dataclasses import dataclass, fields
from pathlib import Path

import jsonschema
import pytest
import yaml

from starfinder.dataset import from_workflow_config
from starfinder.preprocessing import (PREPROCESSING_METHODS, Background3DConfig, MinMaxNormalizationConfig, PercentileNormalizationConfig,
    PreprocessingRecipe, PreprocessingStep, ReconstructionConfig, ScalarBackgroundConfig, StepResult, PreprocessingSpec,
    TophatConfig)

pytestmark = [pytest.mark.preprocessing, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
EXAMPLE = yaml.safe_load((ROOT / "docs/examples/workflow-recipe-2.yaml").read_text())
PYTHON_RULES = ("rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile",
                "deep_rsf_subtile")
LEGACY = {"enhance_contrast": {"run": False}, "hist_equalize": {"run": False}, "morph_recon": {"run": False},
          "tophat": {"run": False}, "snr_threshold": 3.0}
RECIPE_2 = PreprocessingRecipe((PreprocessingStep(ScalarBackgroundConfig(percentile=10.0), save_as="bg_corrected"),
                                PreprocessingStep(PercentileNormalizationConfig(p_low=1.0, p_high=99.9))),
                               extraction_source="bg_corrected")


def with_preprocessing(preprocessing, rule="rsf_single_fov", **extra):
    config = copy.deepcopy(EXAMPLE)
    config["rules"] = {rule: {"run": True, "parameters": {"preprocessing": preprocessing, **extra}}}
    return config


# --- Schema consistency with PREPROCESSING_METHODS -----------------------------------------------------

def schema_steps(schema):
    """Step name -> parameter names declared by the schema's preprocessing step list."""
    defs = schema["$defs"]
    declared = {}
    for option in defs["preprocessing_params"]["properties"]["steps"]["items"]["oneOf"]:
        properties = defs[option["$ref"].rsplit("/", 1)[1]]["properties"]
        declared[properties["method"]["const"]] = set(properties) - {"method", "save_as"}
    return declared


def registered_steps(steps):
    """Step name -> init fields of its config dataclass, for every registered step."""
    return {spec.name: {f.name for f in fields(config_type) if f.init} for config_type, spec in steps.items()}


def schema_mismatches(schema, steps):
    """Differences between the schema's step list and PREPROCESSING_METHODS; empty when they agree."""
    declared, registered = schema_steps(schema), registered_steps(steps)
    problems = [f"step {name!r} is registered but missing from the schema" for name in sorted(set(registered) - set(declared))]
    problems += [f"step {name!r} is in the schema but not registered" for name in sorted(set(declared) - set(registered))]
    problems += [f"step {name!r} parameters differ: schema {sorted(declared[name])}, config {sorted(registered[name])}"
                 for name in sorted(set(declared) & set(registered)) if declared[name] != registered[name]]
    return problems


def test_schema_step_list_matches_steps_and_config_fields():
    assert schema_mismatches(SCHEMA, PREPROCESSING_METHODS) == []


def test_consistency_check_fails_for_a_missing_step_or_different_parameters(monkeypatch):
    missing = copy.deepcopy(SCHEMA)
    missing["$defs"]["preprocessing_params"]["properties"]["steps"]["items"]["oneOf"].pop()
    assert schema_mismatches(missing, PREPROCESSING_METHODS) == ["step 'percentile_normalization' is registered but missing from the schema"]
    extra = copy.deepcopy(SCHEMA)
    extra["$defs"]["preprocessing_step_white_tophat"]["properties"]["radius_z"] = {"type": "integer"}
    assert schema_mismatches(extra, PREPROCESSING_METHODS) == [
        "step 'white_tophat' parameters differ: schema ['radius_yx', 'radius_z'], config ['radius_yx']"]

    @dataclass(frozen=True)
    class GammaConfig:
        gamma: float = 1.0

    monkeypatch.setitem(PREPROCESSING_METHODS, GammaConfig, PreprocessingSpec("gamma_correction", lambda v, c, x: StepResult(v, {}, {}),
                                                     "contrast", "per_channel"))
    assert schema_mismatches(SCHEMA, PREPROCESSING_METHODS) == ["step 'gamma_correction' is registered but missing from the schema"]


# --- Adapter -------------------------------------------------------------------------

def test_recipe_2_example_validates_and_builds_the_declared_recipe():
    jsonschema.validate(EXAMPLE, SCHEMA)
    adapted = from_workflow_config(EXAMPLE, "rsf_single_fov")
    assert adapted.pipeline.preprocessing == RECIPE_2
    assert adapted.pipeline.registration and adapted.pipeline.extraction is not None


@pytest.mark.parametrize("rule", PYTHON_RULES)
def test_every_python_rule_accepts_the_key(rule):
    preprocessing = {"steps": [
        {"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255], "save_as": "scaled"},
        {"method": "background_3d", "radius_voxels_zyx": [1, 3, 3], "save_as": "flat"},
        {"method": "white_tophat", "radius_yx": 2}],
        "extraction_source": "flat", "registration_source": "scaled", "supplied_statistics": None}
    config = with_preprocessing(preprocessing, rule)
    jsonschema.validate(config, SCHEMA)
    recipe = from_workflow_config(config, rule).pipeline.preprocessing
    assert recipe == PreprocessingRecipe((PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255)), "scaled"),
                                          PreprocessingStep(Background3DConfig(radius_voxels_zyx=(1, 3, 3)), "flat"),
                                          PreprocessingStep(TophatConfig(radius_yx=2))),
                                         extraction_source="flat", registration_source="scaled")
    assert recipe.post_registration == ()


def test_supplied_statistics_path_is_passed_to_the_recipe(tmp_path):
    preprocessing = {"steps": [{"method": "percentile_normalization", "fit": "supplied"}],
                     "supplied_statistics": str(tmp_path / "supplied.json")}
    config = with_preprocessing(preprocessing)
    jsonschema.validate(config, SCHEMA)
    assert from_workflow_config(config).pipeline.preprocessing.supplied_statistics == tmp_path / "supplied.json"


# The native form each legacy key's error names (backend python).
NATIVE = {"enhance_contrast": "min_max_normalization", "hist_equalize": "histogram_matching",
          "morph_recon": "reconstruction", "tophat": "white_tophat", "snr_threshold": "min_max_normalization"}


@pytest.mark.parametrize("key", sorted(LEGACY))
def test_each_legacy_key_raises_on_the_python_backend_naming_its_native_form(key):
    """Under backend python a legacy key raises naming its preprocessing step, with or without the preprocessing key,
    and the schema's backend guard refuses it; a MATLAB-backend configuration keeps the mutual exclusion."""
    for config in (with_preprocessing({"steps": [{"method": "white_tophat"}]}, **{key: LEGACY[key]}),
                   with_preprocessing(None, **{key: LEGACY[key]})):
        if config["rules"]["rsf_single_fov"]["parameters"]["preprocessing"] is None:
            del config["rules"]["rsf_single_fov"]["parameters"]["preprocessing"]
        with pytest.raises(ValueError, match=f"{key} is a MATLAB key, which the Python backend replaces by .*"
                                             f"{NATIVE[key]}"):
            from_workflow_config(config)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, SCHEMA)
    matlab = with_preprocessing({"steps": [{"method": "white_tophat"}]}, **{key: LEGACY[key]})
    del matlab["backend"]
    with pytest.raises(ValueError, match="mutually exclusive with the legacy keys"):
        from_workflow_config(matlab)


@pytest.mark.parametrize("key", sorted(LEGACY))
def test_a_legacy_key_of_a_disabled_rule_raises_on_the_python_backend(key):
    """The refusal does not depend on run: a legacy key (here disabled, run false) in a rule that does not run raises
    when any rule is read, naming its native form, and the schema refuses it."""
    config = with_preprocessing({"steps": [{"method": "white_tophat"}]})
    config["rules"]["gr_single_fov_subtile"] = {"run": False, "parameters": {key: LEGACY[key]}}
    with pytest.raises(ValueError, match=f"rules.gr_single_fov_subtile.parameters: {key} is a MATLAB key, which the "
                                         f"Python backend replaces by .*{NATIVE[key]}"):
        from_workflow_config(config)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)


@pytest.mark.parametrize("method", ["reconstruction", "white_tophat"])
@pytest.mark.parametrize("where", ["steps", "post_registration"])
def test_the_matlab_radius_inside_a_native_step_raises_naming_radius_yx(method, where):
    """morph_recon's and tophat's radius inside a native step (before or after registration) names radius_yx."""
    preprocessing = {"steps": [], where: [{"method": method, "radius": 3}]}
    config = with_preprocessing(preprocessing)
    with pytest.raises(ValueError, match=rf"preprocessing.{where}\[0\]: .*radius is the MATLAB key of morph_recon and "
                                         "tophat, which the Python backend writes as radius_yx"):
        from_workflow_config(config)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)


@pytest.mark.parametrize("steps, extra, message", [
    ([{"method": "no_such_step"}], {}, "unknown preprocessing step 'no_such_step'"),
    ([{"radius_yx": 2}], {}, "with a method"),
    ([{"method": "white_tophat", "radius": 2}], {}, "unknown preprocessing step TophatConfig keys"),
    ([{"method": "white_tophat", "save_as": "top"}], {"extraction_source": "missing"}, "extraction_source 'missing' is not a snapshot"),
    ([{"method": "white_tophat"}], {"registration_source": "detection"}, "registration_source 'detection' is not a snapshot"),
    ([{"method": "white_tophat", "save_as": "detection"}], {}, 'must not be "detection"'),
    ([{"method": "white_tophat", "save_as": "a"}, {"method": "white_tophat", "save_as": "a"}], {}, "more than once"),
    ([{"method": "white_tophat"}], {"post_steps": []}, "unknown preprocessing keys"),
    ([{"method": "white_tophat"}], {"post_registration": [{"method": "white_tophat"}]},
     "post_registration may contain only ReconstructionConfig steps"),
    ([{"method": "white_tophat"}], {"post_registration": [{"method": "reconstruction", "save_as": "late"}]},
     "keep no snapshots"),
])
def test_unknown_methods_fields_and_snapshots_raise(steps, extra, message):
    config = with_preprocessing({"steps": steps, **extra})
    with pytest.raises(ValueError, match=message):
        from_workflow_config(config)


@pytest.mark.parametrize("step", [{"method": "no_such_step"}, {"method": "white_tophat", "radius": 2},
                                  {"method": "white_tophat", "save_as": "detection"},
                                  {"method": "percentile_normalization", "fit": "sample"}])
def test_schema_rejects_unknown_methods_fields_and_values(step):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(with_preprocessing({"steps": [step]}), SCHEMA)


@pytest.mark.parametrize("backend", [None, "matlab"])
def test_schema_accepts_the_key_only_on_the_python_backend(backend):
    config = with_preprocessing({"steps": [{"method": "white_tophat"}]})
    config.pop("backend")
    if backend is not None:
        config["backend"] = backend
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    config["backend"] = "python"
    jsonschema.validate(config, SCHEMA)


@pytest.mark.parametrize("rule", PYTHON_RULES)
def test_the_post_registration_list_carries_to_the_recipe(rule):
    """preprocessing.post_registration lists the steps run after registration (reconstruction), in any rule."""
    preprocessing = {"steps": [{"method": "white_tophat", "radius_yx": 2}],
                     "post_registration": [{"method": "reconstruction", "radius_yx": 5}]}
    config = with_preprocessing(preprocessing, rule)
    jsonschema.validate(config, SCHEMA)
    recipe = from_workflow_config(config, rule).pipeline.preprocessing
    assert recipe == PreprocessingRecipe((PreprocessingStep(TophatConfig(radius_yx=2)),),
                                         (PreprocessingStep(ReconstructionConfig(radius_yx=5)),))
    assert from_workflow_config(with_preprocessing({"steps": []}, rule), rule).pipeline.preprocessing \
        .post_registration == ()
    for bad in ([{"method": "white_tophat"}], [{"method": "reconstruction", "save_as": "late"}], {}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(with_preprocessing(dict(preprocessing, post_registration=bad), rule), SCHEMA)


def test_legacy_keys_still_map_to_recipe_1_on_the_matlab_backend():
    """A MATLAB-backend configuration, which the adapters of the shared rules read, keeps the translation of the
    legacy keys (_legacy_config); the Python backend refuses them."""
    config = copy.deepcopy(EXAMPLE)
    parameters = config["rules"]["rsf_single_fov"]["parameters"]
    del parameters["preprocessing"]
    parameters["tophat"] = {"run": True, "radius": 2}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    with pytest.raises(ValueError, match="tophat is a MATLAB key.*white_tophat"):
        from_workflow_config(config)
    del config["backend"]
    recipe = from_workflow_config(config).pipeline.preprocessing
    assert recipe == PreprocessingRecipe((PreprocessingStep(TophatConfig(radius_yx=2)),))
