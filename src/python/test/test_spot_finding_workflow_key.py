"""The spot_finding workflow block: the method key, config fields, the local_maxima legacy aliases,
channel_overrides, the rule-level device, and the static schema's consistency with SPOT_FINDING_METHODS
(W-270, docs/spot-finding-contract.md, "Workflow configuration")."""
import copy
from dataclasses import dataclass, field, fields
from pathlib import Path

import jsonschema
import pytest
import yaml

from starfinder._registry import names
from starfinder.dataset import PipelineConfig
from starfinder.dataset.workflow import from_workflow_config
from starfinder.spot_finding import (SPOT_FINDING_METHODS, ChannelOverride, LocalMaximaConfig, SpotFindingPlan,
    SpotFindingSpec)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
ALIASES = {"intensity_estimation": "threshold_mode", "intensity_threshold": "threshold_value",
           "min_distance": "min_distance_voxels"}


def workflow(spot_finding, **parameters):
    return {"n_rounds": 1, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
            "root_input_path": "in", "root_output_path": "out", "seq_channel_order": list(CHANNELS),
            "rules": {"rsf_single_fov": {"parameters": {"load_raw_images": {"run": False},
                                                        "spot_finding": {"run": True, **spot_finding},
                                                        **parameters}}}}


def detection(spot_finding, **parameters):
    return from_workflow_config(workflow(spot_finding, **parameters)).pipeline.detection


@dataclass(frozen=True)
class NativeDistanceConfig:
    """A fixture pipeline config with a native min_distance field, as Spotiflow and Piscis have."""
    min_distance: int = 1
    threshold: float = 0.5
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="native_distance", init=False)

    def __post_init__(self):
        pass


@pytest.fixture
def native_distance(monkeypatch):
    spec = SpotFindingSpec("native_distance", lambda *a: None, pipeline=True, dimensions=frozenset({2, 3}),
                           output_columns=("z", "y", "x", "channel"))
    monkeypatch.setitem(SPOT_FINDING_METHODS, NativeDistanceConfig, spec)


# --- local_maxima -------------------------------------------------------------------------------------

def test_method_and_config_fields_give_the_python_pipeline():
    block = {"method": "local_maxima", "threshold_mode": "adaptive", "threshold_value": 0.2,
             "min_distance_voxels": 2, "exclude_border": False, "measure_peak_intensity": False}
    adapted = from_workflow_config(workflow(block)).pipeline
    assert adapted == PipelineConfig(detection=LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.2,
        min_distance_voxels=2, exclude_border=False, measure_peak_intensity=False))
    assert detection({"exclude_border": False}) == LocalMaximaConfig(exclude_border=False)
    assert detection({}) == LocalMaximaConfig()


@pytest.mark.parametrize("method", [{}, {"method": "local_maxima"}])
def test_legacy_keys_are_local_maxima_aliases(method):
    block = {**method, "intensity_estimation": "adaptive_round", "intensity_threshold": 0.3, "min_distance": 2}
    assert detection(block) == LocalMaximaConfig(threshold_mode="adaptive_round", threshold_value=0.3,
                                                 min_distance_voxels=2)
    assert detection({**method, "min_distance_voxels": 3}) == LocalMaximaConfig(min_distance_voxels=3)


@pytest.mark.parametrize("alias, name", list(ALIASES.items()))
def test_a_legacy_key_together_with_its_field_raises(alias, name):
    values = {"threshold_mode": "noise", "threshold_value": 4.0, "min_distance_voxels": 2}
    with pytest.raises(ValueError, match=f"{alias} and {name}"):
        detection({alias: values[name], name: values[name]})


@pytest.mark.parametrize("block", [{"method": "noise_landmark"}, {"method": "percentile_centroid"},
                                   {"method": "starfish_log"}, {"method": "local_maxima", "level": 2},
                                   {"channel_labels_typo": ["a"]}])
def test_unknown_or_non_pipeline_methods_and_unknown_keys_raise(block):
    with pytest.raises(ValueError):
        detection(block)


# --- Another method's native min_distance ------------------------------------------------------------

def test_min_distance_is_the_native_field_of_another_method(native_distance):
    assert detection({"method": "native_distance", "min_distance": 3}) == NativeDistanceConfig(min_distance=3)
    assert detection({"method": "native_distance", "threshold": 0.7}) == NativeDistanceConfig(threshold=0.7)


@pytest.mark.parametrize("key, value", [("intensity_estimation", "noise"), ("intensity_threshold", 5.0),
                                        ("min_distance_voxels", 2)])
def test_local_maxima_keys_raise_for_another_method(native_distance, key, value):
    with pytest.raises(ValueError, match=key):
        detection({"method": "native_distance", key: value})


# --- channel_overrides and device ---------------------------------------------------------------------

def test_channel_overrides_build_a_plan():
    block = {"threshold_value": 4.0, "channel_overrides": {
        "ch02": {"threshold_mode": "adaptive", "threshold_value": 0.2}, "ch03": {"intensity_threshold": 6.0}}}
    base = LocalMaximaConfig(threshold_value=4.0)
    assert detection(block) == SpotFindingPlan(base, (
        ChannelOverride("ch02", LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.2)),
        ChannelOverride("ch03", LocalMaximaConfig(threshold_value=6.0))))


@pytest.mark.parametrize("overrides, error", [
    ({"ch09": {"threshold_value": 4.0}}, ValueError), ({"ch02": {"unknown": 1}}, ValueError),
    ({"ch02": {"min_distance": 2, "min_distance_voxels": 2}}, ValueError), (["ch02"], TypeError),
    ({"ch02": 4.0}, TypeError)])
def test_invalid_channel_overrides_raise(overrides, error):
    with pytest.raises(error):
        detection({"channel_overrides": overrides})


def test_the_rule_level_device():
    assert from_workflow_config(workflow({})).execution.device == "cpu"
    assert from_workflow_config(workflow({}, device="cpu")).execution.device == "cpu"
    with pytest.raises(ValueError, match="device must be 'cpu'"):
        from_workflow_config(workflow({}, device="cuda"))


def test_the_shared_config_is_not_mutated():
    config = workflow({"method": "local_maxima", "channel_overrides": {"ch02": {"threshold_value": 4.0}}})
    original = copy.deepcopy(config)
    from_workflow_config(config)
    assert config == original


# --- The static schema --------------------------------------------------------------------------------

def spot_schema():
    return SCHEMA["$defs"]["spot_finding_params"]


def test_schema_method_enum_equals_the_pipeline_methods():
    pipeline = [spec.name for spec in SPOT_FINDING_METHODS.values() if spec.pipeline]
    assert spot_schema()["properties"]["method"]["enum"] == pipeline
    assert set(pipeline) <= set(names(SPOT_FINDING_METHODS))


def test_schema_declares_the_python_only_keys_and_drops_local():
    properties = spot_schema()["properties"]
    assert "local" not in properties["intensity_estimation"]["enum"]
    assert set(properties["intensity_estimation"]["enum"]) == set(properties["threshold_mode"]["enum"])
    # channel_labels comes from seq_channel_order, so the schema does not declare it.
    fields_ = {f.name for config_type, spec in SPOT_FINDING_METHODS.items() if spec.pipeline
               for f in fields(config_type) if f.init and f.name != "channel_labels"}
    assert set(properties) == {"run", "ref_round", "method", "channel_overrides", *ALIASES, *fields_}
    assert SCHEMA["$defs"]["python_rule_parameters"]["properties"]["device"]["enum"] == ["cpu"]


@pytest.mark.parametrize("block, valid", [
    ({"run": True, "method": "local_maxima", "exclude_border": False, "channel_overrides": {"ch02": {
        "threshold_value": 4.0}}}, True),
    ({"run": True, "intensity_estimation": "adaptive", "intensity_threshold": 0.2}, True),
    ({"run": True, "intensity_estimation": "local"}, False),
    ({"run": True, "method": "noise_landmark"}, False),
    ({"run": True, "channel_overrides": {"ch02": 4.0}}, False)])
def test_schema_validates_spot_finding_blocks(block, valid):
    schema = {"$defs": SCHEMA["$defs"], "$ref": "#/$defs/spot_finding_params"}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(block))
    assert (not errors) == valid


EXAMPLE = yaml.safe_load((ROOT / "docs/examples/workflow-recipe-2.yaml").read_text())


@pytest.mark.parametrize("parameters", [
    {"spot_finding": {"run": True, "method": "local_maxima"}},
    {"spot_finding": {"run": True, "channel_overrides": {"ch02": {"threshold_value": 4.0}}}},
    {"device": "cpu"}])
def test_python_only_keys_validate_on_the_python_backend_only(parameters):
    config = copy.deepcopy(EXAMPLE)
    del config["rules"]["rsf_single_fov"]["parameters"]["preprocessing"]
    config["rules"]["rsf_single_fov"]["parameters"]["spot_finding"] = {
        "run": True, "ref_round": "round1", "intensity_estimation": "adaptive", "intensity_threshold": 0.2}
    for backend in ("python", "matlab"):
        jsonschema.validate(dict(config, backend=backend), SCHEMA)  # the shared keys validate on both
    config["rules"]["rsf_single_fov"]["parameters"].update(parameters)
    jsonschema.validate(config, SCHEMA)
    from_workflow_config(config)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(config, backend="matlab"), SCHEMA)
