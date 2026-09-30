"""The Python-only registration workflow key: adapter, static schema and its consistency with REGISTRATION_METHODS (W-256)."""
import copy
from dataclasses import dataclass, field, fields
from pathlib import Path

import jsonschema
import pytest
import yaml

from starfinder.dataset import RecoveryConfig, RegistrationRecipe, RegistrationStep, from_workflow_config
from starfinder.registration import (REGISTRATION_METHODS, AffineConfig, BSplineConfig, DemonsConfig,
    InsufficientLandmarksError, RegistrationEstimationError, RegistrationQcConfig, RegistrationSignalConfig,
    RegistrationSpec, RigidConfig, TranslationConfig, WarpConfig)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
EXAMPLE = yaml.safe_load((ROOT / "docs/examples/workflow-recipe-2.yaml").read_text())
PYTHON_RULES = ("rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile",
                "deep_rsf_subtile")
LEGACY_ALIASES = ("diffeomorphic", "symmetric", "fast_symmetric")
# Step keys of the Python-only key besides the config fields.
STEP_KEYS = {"method", "recovery", "signal"}

EXPLICIT = """
signal: {mode: channel, reference_channel: ch00, moving_channel: 1}
warp: {boundary_mode: nearest, backend: scipy}
qc: {min_coverage: 0.5, max_fold_fraction: null}
steps:
  - method: translation
    backend: skimage
  - method: affine
    iterations: 50
    recovery: {allowed_errors: [RegistrationEstimationError], alternatives: [{method: rigid}]}
  - method: bspline
    grid_spacing_physical: 8.0
    signal: {mode: max}
    recovery:
      allowed_errors: [RegistrationEstimationError, InsufficientLandmarksError]
      alternatives: [{method: demons, iterations: [10, 5]}]
"""
EXPLICIT_RECIPE = RegistrationRecipe(
    (RegistrationStep(TranslationConfig(backend="skimage")),
     RegistrationStep(AffineConfig(iterations=50), RecoveryConfig((RegistrationEstimationError,), (RigidConfig(),))),
     RegistrationStep(BSplineConfig(grid_spacing_physical=8.0),
                      RecoveryConfig((RegistrationEstimationError, InsufficientLandmarksError),
                                     (DemonsConfig(iterations=(10, 5)),)),
                      RegistrationSignalConfig("max"))),
    signal=RegistrationSignalConfig("channel", "ch00", 1), warp=WarpConfig(backend="scipy", boundary_mode="nearest"),
    qc=RegistrationQcConfig(min_coverage=0.5))


def with_registration(registration, rule="rsf_single_fov", **extra):
    config = copy.deepcopy(EXAMPLE)
    config["rules"] = {rule: {"run": True, "parameters": {"registration": registration, **extra}}}
    return config


# --- Schema consistency with REGISTRATION_METHODS ---------------------------------------------

def schema_methods(schema):
    """Method name -> parameter names declared by the schema's registration step list."""
    defs = schema["$defs"]
    declared = {}
    for option in defs["registration_params"]["properties"]["steps"]["items"]["oneOf"]:
        properties = defs[option["$ref"].rsplit("/", 1)[1]]["properties"]
        declared[properties["method"]["const"]] = set(properties) - STEP_KEYS
    return declared


def registered_methods(methods):
    """Method name -> init fields of its config dataclass, for every registered method."""
    return {spec.name: {f.name for f in fields(config_type) if f.init} for config_type, spec in methods.items()}


def schema_mismatches(schema, methods):
    """Differences between the schema and REGISTRATION_METHODS and the recipe option types; empty when they agree."""
    declared, registered = schema_methods(schema), registered_methods(methods)
    problems = [f"method {name!r} is registered but missing from the schema" for name in sorted(set(registered) - set(declared))]
    problems += [f"method {name!r} is in the schema but not registered" for name in sorted(set(declared) - set(registered))]
    problems += [f"method {name!r} parameters differ: schema {sorted(declared[name])}, config {sorted(registered[name])}"
                 for name in sorted(set(declared) & set(registered)) if declared[name] != registered[name]]
    defs = schema["$defs"]
    for name, cls in (("registration_signal", RegistrationSignalConfig), ("registration_warp", WarpConfig),
                      ("registration_qc", RegistrationQcConfig)):
        expected = {f.name for f in fields(cls) if f.init}
        if set(defs[name]["properties"]) != expected:
            problems.append(f"{name} differs: schema {sorted(defs[name]['properties'])}, config {sorted(expected)}")
    recipe = {f.name for f in fields(RegistrationRecipe)}
    if set(defs["registration_params"]["properties"]) != recipe:
        problems.append(f"registration_params differs from RegistrationRecipe {sorted(recipe)}")
    local = {spec.name for spec in methods.values() if spec.step_kind == "local"} | set(LEGACY_ALIASES)
    enum = set(defs["local_registration_params"]["properties"]["method"]["enum"])
    if enum != local:
        problems.append(f"local_registration.method enum {sorted(enum)} differs from {sorted(local)}")
    return problems


def test_schema_registration_methods_match_the_registry_and_config_fields():
    assert schema_mismatches(SCHEMA, REGISTRATION_METHODS) == []


def test_consistency_check_fails_for_a_missing_method_or_different_parameters(monkeypatch):
    missing = copy.deepcopy(SCHEMA)
    missing["$defs"]["registration_params"]["properties"]["steps"]["items"]["oneOf"].pop()
    assert schema_mismatches(missing, REGISTRATION_METHODS) == ["method 'cpd' is registered but missing from the schema"]
    extra = copy.deepcopy(SCHEMA)
    extra["$defs"]["registration_step_translation"]["properties"]["upsample"] = {"type": "integer"}
    extra["$defs"]["registration_warp"]["properties"].pop("fft_workers")
    extra["$defs"]["local_registration_params"]["properties"]["method"]["enum"].remove("bspline")
    assert schema_mismatches(extra, REGISTRATION_METHODS) == [
        "method 'translation' parameters differ: schema ['backend', 'fft_workers', 'upsample'], config ['backend', 'fft_workers']",
        "registration_warp differs: schema ['backend', 'boundary_mode', 'clip_to_dtype', 'fill_value', 'integer_rounding', "
        "'output_dtype'], config ['backend', 'boundary_mode', 'clip_to_dtype', 'fft_workers', 'fill_value', "
        "'integer_rounding', 'output_dtype']",
        "local_registration.method enum ['cpd', 'demons', 'diffeomorphic', 'fast_symmetric', 'symmetric', 'tps'] differs "
        "from ['bspline', 'cpd', 'demons', 'diffeomorphic', 'fast_symmetric', 'symmetric', 'tps']"]

    @dataclass(frozen=True)
    class FixtureLocalConfig:
        weight: float = 1.0
        method: str = field(default="fixture_local", init=False)

    monkeypatch.setitem(REGISTRATION_METHODS, FixtureLocalConfig, RegistrationSpec(
        "fixture_local", lambda *a: None, step_kind="local", dimensions=frozenset({3}), transform_kind="dense",
        space="index"))
    assert schema_mismatches(SCHEMA, REGISTRATION_METHODS) == [
        "method 'fixture_local' is registered but missing from the schema",
        "local_registration.method enum ['bspline', 'cpd', 'demons', 'diffeomorphic', 'fast_symmetric', 'symmetric', "
        "'tps'] differs from ['bspline', 'cpd', 'demons', 'diffeomorphic', 'fast_symmetric', 'fixture_local', "
        "'symmetric', 'tps']"]


def test_matlab_registration_keys_keep_their_names():
    defs = SCHEMA["$defs"]
    assert set(defs["global_registration_params"]["properties"]) == {"run", "ref_round", "ref_img", "mov_img",
                                                                      "ref_channel", "recovery"}
    assert defs["global_registration_params"]["properties"]["ref_img"]["enum"] == ["merged-image", "single-channel"]
    # The local block declares ref_img and mov_img as Python-only keys (W-246 amendment).
    assert set(defs["local_registration_params"]["properties"]) == {"run", "ref_round", "method", "ref_img", "mov_img",
                                                                     "ref_channel", "recovery"}
    assert defs["local_registration_params"]["properties"]["method"]["default"] == "demons"


# --- Adapter ---------------------------------------------------------------------------------

@pytest.mark.parametrize("rule", PYTHON_RULES)
def test_every_python_rule_accepts_the_key(rule):
    config = with_registration(yaml.safe_load(EXPLICIT), rule)
    original = copy.deepcopy(config)
    jsonschema.validate(config, SCHEMA)
    assert from_workflow_config(config, rule).pipeline.registration == EXPLICIT_RECIPE
    assert config == original


def test_minimal_key_and_defaults():
    config = with_registration(yaml.safe_load("steps: [{method: translation}, {method: demons, iterations: [5]}]"))
    jsonschema.validate(config, SCHEMA)
    recipe = from_workflow_config(config).pipeline.registration
    assert recipe == RegistrationRecipe((RegistrationStep(TranslationConfig()),
                                         RegistrationStep(DemonsConfig(iterations=(5,)))))
    assert recipe.signal == RegistrationSignalConfig("max") and recipe.warp is None


@pytest.mark.parametrize("block", ["global_registration", "local_registration"])
def test_the_key_is_mutually_exclusive_with_an_enabled_legacy_block(block):
    config = with_registration({"steps": [{"method": "translation"}]}, **{block: {"run": True}})
    config["rules"]["rsf_single_fov"]["parameters"].pop("global_registration", None)
    config["rules"]["rsf_single_fov"]["parameters"][block] = {"run": True}
    with pytest.raises(ValueError, match="mutually exclusive with the enabled legacy blocks"):
        from_workflow_config(config)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    config["rules"]["rsf_single_fov"]["parameters"][block] = {"run": False}
    jsonschema.validate(config, SCHEMA)
    assert from_workflow_config(config).pipeline.registration == RegistrationRecipe((RegistrationStep(TranslationConfig()),))


@pytest.mark.parametrize("registration, error, message", [
    ({"steps": [{"method": "no_such_method"}]}, ValueError, "unknown registration method 'no_such_method'"),
    ({"steps": [{"iterations": 5}]}, ValueError, "with a method"),
    ({"steps": []}, ValueError, "at least one step"),
    ({"steps": [{"method": "translation", "upsample": 2}]}, ValueError, "unknown registration step TranslationConfig keys"),
    ({"steps": [{"method": "demons"}, {"method": "translation"}]}, ValueError, "global steps followed by at most one local"),
    ({"steps": [{"method": "diffeomorphic"}]}, ValueError, "unknown registration method 'diffeomorphic'"),
    ({"steps": [{"method": "tps", "recovery": {"allowed_errors": ["InsufficientLandmarksError"],
                                               "alternatives": [{"method": "affine"}]}}]}, ValueError, "is not a local method"),
    ({"steps": [{"method": "tps", "recovery": {"allowed_errors": ["InsufficientLandmarksError"],
                                               "alternatives": [{"method": "cpd", "signal": {"mode": "sum"}}]}}]},
     ValueError, "configure estimators only"),
    ({"steps": [{"method": "translation"}], "signal": {"mode": "mean"}}, ValueError, "mode must be one of"),
    ({"steps": [{"method": "translation"}], "warp": {"backend": "scipy"}}, ValueError, 'requires warp backend "translation"'),
    ({"steps": [{"method": "translation"}], "stages": []}, ValueError, "unknown registration keys"),
    ({"steps": [{"method": "translation"}], "reference_round": "round2"}, ValueError, "differs from dataset reference"),
    ({"steps": [{"method": "translation"}], "qc": [0.5]}, TypeError, "registration qc must be a mapping"),
])
def test_unknown_methods_fields_and_sequences_raise(registration, error, message):
    with pytest.raises(error, match=message):
        from_workflow_config(with_registration(registration))


@pytest.mark.parametrize("registration", [
    {"steps": [{"method": "no_such_method"}]},
    {"steps": [{"method": "translation", "upsample": 2}]},
    {"steps": []},
    {"steps": [{"method": "translation"}], "signal": {"mode": "mean"}},
    {"steps": [{"method": "translation"}], "qc": {"min_coverage": 2}},
    {"steps": [{"method": "tps", "recovery": {"allowed_errors": ["ValueError"], "alternatives": [{"method": "cpd"}]}}]},
])
def test_schema_rejects_unknown_methods_fields_and_values(registration):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(with_registration(registration), SCHEMA)


@pytest.mark.parametrize("backend", [None, "matlab"])
def test_schema_accepts_the_key_only_on_the_python_backend(backend):
    config = with_registration({"steps": [{"method": "translation"}]})
    config.pop("backend")
    if backend is not None:
        config["backend"] = backend
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    config["backend"] = "python"
    jsonschema.validate(config, SCHEMA)


def test_the_widened_local_method_enum_is_accepted():
    for method in sorted({s.name for s in REGISTRATION_METHODS.values() if s.step_kind == "local"} | set(LEGACY_ALIASES)):
        config = copy.deepcopy(EXAMPLE)
        config["rules"]["rsf_single_fov"]["parameters"]["local_registration"] = {"run": True, "method": method}
        jsonschema.validate(config, SCHEMA)
        recipe = from_workflow_config(config).pipeline.registration
        assert REGISTRATION_METHODS[type(recipe.steps[-1].config)].step_kind == "local"
