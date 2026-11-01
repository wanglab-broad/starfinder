"""RS5 of design/validation/sample-segmentation.md: the route in the workflow (W-430).

RS5: the static schema's ``export`` keys equal the fields of ``ExportConfig`` with ``route``,
``segment`` and ``segmentation``, and a run's keys besides ``name``, ``target``, ``inputs``,
``seeds`` and ``method`` equal the init fields of its method's config, kept equal to
``SEGMENTATION_METHODS`` as the §2.9 schema test does; the configurations of each route, and the
example configurations of each route on the workflow configuration page, pass the package's
validation (the Python schema, the translators and the data dependencies, as the plan validates
them; W-750) and translate. WF1 (the inputs of the export task by route, the refusal of the route)
moved to the plan's declarations and refusals, WF2 (the assignment figures) to the export command's
previews: test_workflow_declared_outputs.py, test_workflow_plan_refusals.py and
test_workflow_export_tasks.py (W-750).

Fixtures: ``route_configs`` (workflow ``export`` blocks for every combination of ``route``,
``dimensionality``, ``segment``, ``fuse``, with and without ``segmentation``, and ``assignment``
set or null), as the translator reads them (the repository's full example configuration with the
blocks) and in the Python shape. No model is loaded or run.
"""
import copy
import importlib.util
import itertools
import re
from dataclasses import asdict, fields
from pathlib import Path

import jsonschema
import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder._registry import names
from starfinder.export import (
    ROUTES,
    ExportConfig,
    FusionConfig,
    LabelConfig,
    SampleSegmentationConfig,
    SampleSegmentationRun,
    SpatialDataExportConfig,
)
from starfinder.segmentation import SEGMENTATION_METHODS, LabelImportConfig
from starfinder.segmentation._methods import ROLES

from . import docs_pages
from .sample_segmentation_fixtures import (
    length_configs,
)

pytestmark = [pytest.mark.workflow, pytest.mark.validation]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
EXPORT_BLOCK = SCHEMA["$defs"]["export_block"]
VALIDATOR = jsonschema.validators.validator_for(SCHEMA)(SCHEMA)  # checked once, reused for every configuration
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 1, "order": "column_major_down_right",
                                            "overlap": 0.25}}}
# The images of the route_configs blocks: the four kinds of source entry, one of them an absolute pattern.
ROUTE_IMAGES = ["reference_stain", "reference_merged", "dapi=file:images/dapi/{fov_id}.tif",
                "flamingo=file:/abs/flamingo/{fov_id}.tif"]
SEGMENTATION = {"device": "cpu", "cells": "cell", "nuclei": "nucleus", "runs": [
    {"name": "nucleus", "target": "nucleus", "method": "import", "path": "/in/nuclei.ome.tif"},
    {"name": "cell", "target": "cell", "inputs": [{"role": "cytoplasm", "image": "flamingo"}], "seeds": "nucleus",
     "method": "seeded_watershed", "sigma_um": 1.0}]}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def base_config(**blocks):
    """The repository's full example configuration on the Python backend with a stitching block and the given
    blocks."""
    config = yaml.safe_load((ROOT / "src/python/test/data/workflow_examples/workflow-full.yaml").read_text())
    return {**config, "backend": "python", "stitching": copy.deepcopy(MULTIVIEW), **copy.deepcopy(blocks)}


# --- route_configs -------------------------------------------------------------------------------------------

ROUTE_VALUES = (None, "per_fov", "after_fusion")
DIMENSIONALITIES = ("projection", "stack", "both")
COMBINATIONS = list(itertools.product(ROUTE_VALUES, DIMENSIONALITIES, (True, False), (True, False), (True, False),
                                      ("default", None)))


def route_block(route, dimensionality, segment, fuse, segmentation):
    block = {"images": list(ROUTE_IMAGES), "dimensionality": dimensionality, "segment": segment, "fuse": fuse}
    if route is not None:
        block["route"] = route
    if segmentation:
        block["segmentation"] = copy.deepcopy(SEGMENTATION)
    return block


def route_workflow_config(route, dimensionality, segment, fuse, segmentation, assignment):
    return base_config(assembly={"assignment": assignment}, export=route_block(route, dimensionality, segment, fuse,
                                                                              segmentation))


def python_route_config(tmp_path, route, dimensionality, segment, fuse, segmentation, assignment):
    """The route_configs block in the Python shape (the worked example's dataset keys, processing and morphology
    blocks, the stitching block); an assignment name gets the assignment block and the segment run it reads."""
    from . import workflow_examples as X
    block = route_block(route, dimensionality, segment, fuse, segmentation)
    config = dict(X.dataset(tmp_path, "route"), processing=copy.deepcopy(X.PROCESSING), morphology={},
                  stitching=copy.deepcopy(MULTIVIEW), assembly={"assignment": assignment}, export=block)
    if assignment is not None:
        config["segmentation"] = {"runs": [{"name": "cell", "target": "cell", "method": "import",
                                            "path": str(tmp_path / "{fov_id}.tif")}]}
        config["assignment"] = {"cells": "cell", "name": assignment}
    return config


def resolved(route, dimensionality):
    """The route of the contract's rule, by hand: the set value, else after_fusion for projection only."""
    return route or ("after_fusion" if dimensionality == "projection" else "per_fov")


def refused(route, dimensionality, segment, fuse, segmentation, assignment):
    """Whether check 7 refuses the configuration alone (the reuse conditions and the models aside)."""
    if resolved(route, dimensionality) != "after_fusion":
        return None
    if dimensionality == "stack":
        return "dimensionality 'stack'"
    if assignment is None:
        return None
    if not segmentation:
        return r"export\.segmentation"
    if segment and not fuse:
        return "fuse: false"
    return None


# --- RS5: configuration schema -------------------------------------------------------------------------------

def test_rs5_configuration_schema():
    """The export keys, the route values, the segmentation keys, a run's keys and the method list."""
    declared = EXPORT_BLOCK["properties"]
    nested = {"assembly", "table_format", "fusion", "labels", "spatialdata"}
    expected = ({f.name for f in fields(ExportConfig) if f.name not in nested}
                | {f.name for f in fields(FusionConfig) if f.init}
                | {f"labels_{f.name}" for f in fields(LabelConfig) if f.init}
                | {f"spatialdata_{f.name}" for f in fields(SpatialDataExportConfig) if f.init})
    assert set(declared) == expected == set(workflow._export_keys())
    assert {"route", "segment", "segmentation"} <= set(declared)
    assert declared["route"]["enum"] == [*ROUTES, None]
    assert declared["segment"]["type"] == "boolean"
    # The segmentation sub-block: the SampleSegmentationConfig fields; its assignment is the assignment block's.
    block = declared["segmentation"]
    assert block["type"] == ["object", "null"] and block["additionalProperties"] is False
    assert set(block["properties"]) == {f.name for f in fields(SampleSegmentationConfig)} - {"assignment"}
    assert block["properties"]["device"]["enum"] == ["cpu", "cuda"]
    run = block["properties"]["runs"]["items"]
    assert set(run["properties"]) == set(workflow._SAMPLE_SEGMENTATION_RUN_KEYS) \
        == {f.name for f in fields(SampleSegmentationRun)}
    assert run["additionalProperties"] is True  # the method's config fields, validated by the adapter
    methods = run["properties"]["method"]["anyOf"]
    assert set(methods[0]["enum"]) == set(names(SEGMENTATION_METHODS)) and len(methods[0]["enum"]) == len(
        set(methods[0]["enum"]))
    assert methods[1] == {"const": "import"}
    item = run["properties"]["inputs"]["items"]
    assert set(item["properties"]) == {"role", "image"} and item["properties"]["role"]["enum"] == list(ROLES)


@pytest.mark.parametrize("method", [*names(SEGMENTATION_METHODS), "import"])
def test_rs5_a_run_s_method_keys_equal_the_config_fields(method, monkeypatch):
    """The keys the adapter accepts beside the run keys are the init fields of the method's config (of
    LabelImportConfig without its target, which is the run's, for import)."""
    seen = []
    real = workflow._known
    monkeypatch.setattr(workflow, "_known", lambda values, allowed, context: seen.append(set(allowed)) or
                        real(values, allowed, context))
    with pytest.raises((ValueError, TypeError)):
        workflow._run_method({"not_a_field": 1}, "run", method, "cell", "export.segmentation run")
    if method == "import":
        expected = {f.name for f in fields(LabelImportConfig) if f.init} - {"target"}
    else:
        config_type = next(t for t, spec in SEGMENTATION_METHODS.items() if spec.name == method)
        expected = {f.name for f in fields(config_type) if f.init}
    assert seen == [expected]


def test_rs5_every_init_field_of_a_method_config_translates(tmp_path):
    """length_configs and an import, each written as a run with every init field, give back the same config."""
    configs = dict(length_configs(tmp_path), imported=LabelImportConfig(str(tmp_path / "n.tif"), "nucleus"))
    runs = []
    for key, config in configs.items():
        values = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()}
        if key == "imported":
            runs.append({"name": "nucleus", "target": "nucleus", "method": "import",
                         **{k: v for k, v in values.items() if k != "target"}})
            continue
        spec = SEGMENTATION_METHODS[type(config)]
        target = "nucleus" if "nucleus" in spec.targets else "cell"
        role = sorted(spec.roles)[0]
        runs.append({"name": key, "target": target, "inputs": [{"role": role, "image": "dapi"}], "method": spec.name,
                     **({"seeds": "nucleus"} if spec.seeds == "required" else {}), **values})
    runs.insert(0, runs.pop())  # the import first: the seeds of the watershed
    block = {"images": ["dapi=file:images/dapi/{fov_id}.tif"], "route": "after_fusion",
             "segmentation": {"cells": "seeded_watershed", "nuclei": "nucleus", "runs": runs}}
    config = base_config(assembly={"assignment": "default"}, export=block)
    from . import workflow_tasks as T
    python = python_route_config(tmp_path, "after_fusion", "projection", True, True, False, "default")
    assert T.validation_error(dict(python, export=block)) is None
    export, _ = workflow._export_config(config)
    expected = {"nucleus": configs.pop("imported"), **configs}
    assert {run.name: run.method for run in export.segmentation.runs} == expected


@pytest.mark.parametrize("combination", COMBINATIONS, ids=lambda c: "-".join(map(str, c)))
def test_rs5_route_configs_validate_and_translate(combination, tmp_path):
    """Every route_configs block passes the package's validation, or is refused there naming the key the route's
    check names (check 7 of the contract, which the old schema did not hold; W-750), and translates to the
    ExportConfig of the same values; the segmentation's assignment settings are the assignment block's."""
    from . import workflow_tasks as T
    route, dimensionality, segment, fuse, segmentation, assignment = combination
    error = T.validation_error(python_route_config(tmp_path, *combination))
    expected = refused(*combination)
    assert (error is None) if expected is None else re.search(expected, error), (expected, error)
    config = route_workflow_config(*combination)
    export, block = workflow._export_config(config)
    assert block.assignment == assignment
    assert (export.route, export.segment, export.fuse, export.fusion.dimensionality) == (route, segment, fuse,
                                                                                         dimensionality)
    assert export.fusion.images == tuple(ROUTE_IMAGES)
    if not segmentation:
        assert export.segmentation is None
        return
    nucleus, cell = export.segmentation.runs
    assert (nucleus.name, nucleus.target, nucleus.inputs, nucleus.method) == (
        "nucleus", "nucleus", (), LabelImportConfig("/in/nuclei.ome.tif", "nucleus"))
    assert (cell.inputs, cell.seeds, cell.method.sigma_um) == ((("cytoplasm", "flamingo"),), "nucleus", 1.0)
    assert (export.segmentation.cells, export.segmentation.nuclei, export.segmentation.device) == (
        "cell", "nucleus", "cpu")


@pytest.mark.parametrize("change", [
    {"route": "per_sample"},
    {"route": "after-fusion"},
    {"segment": "no"},
    {"segmentation": dict(SEGMENTATION, device="gpu")},
    {"segmentation": dict(SEGMENTATION, expansion=1.0)},              # an assignment setting: the assignment block's
    {"segmentation": dict(SEGMENTATION, runs=[])},
    {"segmentation": dict(SEGMENTATION, runs=[{"name": "cell", "target": "cell", "method": "watershed"}])},
])
def test_rs5_invalid_route_keys_fail_the_schema(change, tmp_path):
    """The package refuses each (the Python schema or the export block's translator, as the plan validates it;
    W-750)."""
    from . import workflow_tasks as T
    config = python_route_config(tmp_path, None, "projection", True, True, True, "default")
    config["export"].update(copy.deepcopy(change))
    assert T.validation_error(config)


#: A cellpose run that names its diameter and model, so that a run's own invalid key is what the package refuses.
CELLPOSE = {"name": "cell", "method": "cellpose", "diameter": 30.0, "model": "cpsam_v2"}


@pytest.mark.parametrize("run, message", [
    (dict(CELLPOSE, target="membrane"), "target must be nucleus or cell; got 'membrane"),
    (dict(CELLPOSE, target="cell", inputs=[{"role": "dapi", "image": "dapi"}]),
     "run 'cell': unknown channel role 'dapi'; roles are nuclear, cytoplasm, membrane, amplicon, composite"),
    (dict(CELLPOSE, target="cell", inputs=[{"role": "nuclear"}]),
     "run 'cell': each input is a (role, image) pair of strings; got ('nuclear', None)"),
])
def test_rs5_an_invalid_run_key_hits_its_own_refusal(run, message, tmp_path):
    """From test_rs5_invalid_route_keys_fail_the_schema's three runs (target membrane, role dapi, an input with no
    image): with the diameter and model cellpose needs, which the package refuses first without them, each is refused
    by its own key's message."""
    from . import workflow_tasks as T
    config = python_route_config(tmp_path, None, "projection", True, True, True, "default")
    config["export"].update(segmentation=dict(SEGMENTATION, runs=[run]))
    assert T.validation_error(config) == message


def page_examples():
    """The preserved workflow configuration examples that set export.route."""
    text = docs_pages.design("specs/workflow-configuration.md")
    blocks = [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]
    return [b for b in blocks if isinstance(b, dict) and isinstance(b.get("export"), dict)
            and "route" in b["export"]]


@pytest.mark.contract
def test_rs5_the_example_configuration_of_each_route():
    """design/specs/workflow-configuration.md gives one example per route; each validates, translates and passes the
    parse-time check of the route. It reads the page, so it is ``contract`` and runs in Q: this module holds no
    ``slow``, ``learned`` or ``extended`` test (docs/contributing.md, "Test markers"; W-602)."""
    examples = page_examples()
    assert sorted(e["export"]["route"] for e in examples) == ["after_fusion", "per_fov"]
    from . import workflow_tasks as T
    for example in examples:
        config = base_config(**example)
        python = python_route_config(Path("/work"), None, "projection", True, True, False, "default")
        python.update(copy.deepcopy(example))
        assert T.validation_error(python) is None, example
        workflow._check_export_route(config)
        export, block = workflow._export_config(config)
        assert workflow._after_fusion(export, block) is (example["export"]["route"] == "after_fusion")


# --- WF1: the rule's inputs by route ---------------------------------------------------------------------------

# --- WF2: the assignment figures --------------------------------------------------------------------------------
