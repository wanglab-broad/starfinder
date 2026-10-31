"""The MATLAB rule set's configuration validation (W-570).

``workflow/rules/legacy.smk`` (the MATLAB rule set, which the dispatcher includes for ``backend: matlab``) calls
``snakemake.utils.validate(config, schema=...)``, which inserts every schema default into the configuration before it
checks it (``set_default=True``). The MATLAB schema gives the keys that the Python dialect refuses no default, and
``common.smk`` applies MATLAB's defaults instead. The Python workflow validates its configuration with the package
(the plan's Python schema, without inserting defaults; W-750): a Python configuration with a key of the MATLAB
dialect is refused there, naming the key.
"""

import copy
from pathlib import Path

import jsonschema
import pytest
import yaml

from . import workflow_tasks as T
from .test_fov_membership import common_names

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "workflow" / "schemas" / "config.schema.yaml"
SCHEMA = yaml.safe_load(SCHEMA_PATH.read_text())
# The keys the Python backend refuses (the schema's first allOf clause), by group.
REFUSED = {"naming": {"fov_id_pattern": "tile_{i}", "n_fovs": 1}, "list": {"subset_list": []},
           "range": {"subset_range": False, "subset_start": 1, "subset_end": 1},
           "random": {"subset_random": False, "n_random_tests": 1}}
# The subset switches whose default the schema held on the base revision (false), now applied by common.smk.
BASE_DEFAULTS = {"subset_range": False, "subset_random": False}


def default_validator(schema):
    """Snakemake's validator: the schema's validator class whose ``properties`` keyword first inserts each property's
    default into the instance (snakemake 7.32.4 ``utils.validate``, after the jsonschema FAQ's recipe)."""
    cls = jsonschema.validators.validator_for(schema)
    validate_properties = cls.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for name, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(name, subschema["default"])
        yield from validate_properties(validator, properties, instance, schema)

    return jsonschema.validators.extend(cls, {"properties": set_defaults})(schema)


VALIDATOR = default_validator(SCHEMA)


def snakefile_errors(config, validator=VALIDATOR):
    """The validation errors of the Snakefile's call on ``config``, which it updates in place with the defaults."""
    return list(validator.iter_errors(config))


@pytest.mark.parametrize("key, value", [(k, v) for group in REFUSED.values() for k, v in group.items()])
def test_a_python_configuration_with_a_refused_key_fails_naming_the_key(key, value):
    """The package refuses the key in a Python configuration, naming it and where it went (W-750: the Snakefile's
    validation, whose guard named the key, no longer reads a Python configuration)."""
    config = T.dataset_keys("/work", processing={"reads_filtration": {}})
    assert T.validation_error(config) is None
    assert T.validation_error(dict(config, **{key: value})).startswith(f"{key} left the Python configuration")


def test_no_refused_key_has_a_schema_default():
    refused = set().union(*REFUSED.values())
    python_guard = SCHEMA["allOf"][0]["then"]["not"]["anyOf"]
    assert {branch["required"][0] for branch in python_guard} == refused

    def defaults(node):
        if isinstance(node, dict):
            for name, subschema in (node.get("properties") or {}).items():
                if name in refused and isinstance(subschema, dict) and "default" in subschema:
                    yield name
            for child in node.values():
                yield from defaults(child)
        elif isinstance(node, list):
            for child in node:
                yield from defaults(child)

    assert list(defaults(SCHEMA)) == []


def base_schema():
    """The schema of the base revision: the subset switches with their former default."""
    schema = copy.deepcopy(SCHEMA)
    for key, value in BASE_DEFAULTS.items():
        schema["properties"][key]["default"] = value
    return schema


def matlab_configs(tmp_path):
    """The repository's MATLAB configurations, as written and without the subset switches, under tmp_path."""
    tissue = yaml.safe_load((ROOT / "tests" / "tissue_2D_test.yaml").read_text())
    minimal = yaml.safe_load((ROOT / "tests" / "minimal_config.yaml").read_text())
    minimal.update(subset_list=[], subset_start=1, subset_end=1, n_random_tests=1)
    configs = {"tissue_2D_test": tissue,
               "tissue_2D_test-switches": {k: v for k, v in tissue.items() if k not in BASE_DEFAULTS},
               "minimal_config": yaml.safe_load((ROOT / "tests" / "minimal_config.yaml").read_text()),
               "minimal_config-switches": minimal}
    for config in configs.values():
        assert config.get("backend", "matlab") == "matlab"
        config["root_output_path"] = str(tmp_path)
    return configs


@pytest.mark.parametrize("name", ["tissue_2D_test", "tissue_2D_test-switches", "minimal_config",
                                  "minimal_config-switches"])
def test_a_matlab_configuration_passes_and_holds_the_base_subset_values(tmp_path, monkeypatch, name):
    config = matlab_configs(tmp_path)[name]
    base = copy.deepcopy(config)
    assert snakefile_errors(base, default_validator(base_schema())) == []
    assert snakefile_errors(config) == []
    annotation = tmp_path / config["dataset_id"] / config["output_id"] / "documents" / "sample-annotation.csv"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(f"sample_id,fov_start,fov_end\ns1,1,{config['n_fovs']}\n")
    common_names(config, monkeypatch)
    subset = [k for k in base if k.startswith("subset_") or k == "n_random_tests"]
    assert {k: config[k] for k in subset} == {k: base[k] for k in subset}
    assert set(BASE_DEFAULTS) <= set(subset)


@pytest.mark.parametrize("switch", ["subset_range", "subset_random"])
def test_a_matlab_switch_set_to_false_needs_no_companions(tmp_path, switch):
    config = dict(matlab_configs(tmp_path)["minimal_config"], **{switch: False})
    assert not {"subset_start", "subset_end", "n_random_tests"} & set(config)
    assert snakefile_errors(config) == []


@pytest.mark.parametrize("switch, companions", [("subset_range", ["subset_start", "subset_end"]),
                                                ("subset_random", ["n_random_tests"])])
def test_a_matlab_switch_set_to_true_fails_naming_its_companions(tmp_path, switch, companions):
    config = dict(matlab_configs(tmp_path)["minimal_config"], **{switch: True})
    errors = snakefile_errors(config)
    assert [e.validator for e in errors] == ["required"] * len(companions)
    assert [e.message for e in errors] == [f"{key!r} is a required property" for key in companions]


@pytest.mark.parametrize("name, fovs", [("tissue_2D_test", ["tile_1"]), ("minimal_config", ["tile_1"])])
def test_common_smk_selects_the_matlab_fovs(tmp_path, monkeypatch, name, fovs):
    """The MATLAB branch's FOVS after validation: tissue_2D_test's subset_list (as on the base revision), and every
    one of the plain template's n_fovs FOVs."""
    config = matlab_configs(tmp_path)[name]
    assert snakefile_errors(config) == []
    annotation = tmp_path / config["dataset_id"] / config["output_id"] / "documents" / "sample-annotation.csv"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(f"sample_id,fov_start,fov_end\ns1,1,{config['n_fovs']}\n")
    assert common_names(config, monkeypatch)["FOVS"] == fovs
    if name == "minimal_config":
        assert fovs == [config["fov_id_pattern"].format(i=i) for i in range(1, config["n_fovs"] + 1)]


#: Python-only keys that the MATLAB schema refuses with backend matlab (its Python-only clause; OC5 for the inputs).
PYTHON_ONLY = {"selected_fovs": ["tile_1"], "reference_stains": [{"channel": "ch04", "name": "DAPI"}],
               "codebook": "panel.csv", "direct_panel": "panel.csv", "image_dir_template": "{fov}/{round}"}


def test_the_validator_is_snakemakes(tmp_path):
    """With Snakemake installed (it is not a test dependency; the run's saved comparison uses the real one), its own
    validate agrees with the validator above on the repository's MATLAB configurations (it inserts the same defaults)
    and refuses, naming each, the Python-only keys in a MATLAB configuration (W-750: the MATLAB rule set is the
    Snakefile's validation's only reader)."""
    utils = pytest.importorskip("snakemake.utils", reason="snakemake is not installed in the test environment")
    configs = matlab_configs(tmp_path)
    for name, config in configs.items():
        ours = copy.deepcopy(config)
        utils.validate(config, schema=str(SCHEMA_PATH))
        assert snakefile_errors(ours) == [] and config == ours, name
    for key, value in PYTHON_ONLY.items():
        config = dict(configs["minimal_config"], backend="matlab", **{key: value})
        assert snakefile_errors(copy.deepcopy(config)), key
        with pytest.raises(Exception, match="not"):
            utils.validate(config, schema=str(SCHEMA_PATH))
