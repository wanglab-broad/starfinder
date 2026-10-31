"""WF1 of design/specs/report-algorithms.md: the workflow's ``report`` block (W-458; design/specs/report.md,
"Workflow configuration", option WR1).

The ``report`` block: its keys in the MATLAB schema's definitions (``workflow/schemas/config.schema.yaml``, which
refuses the block without ``backend: python``), its validation by the package (the Python schema and the block
translator, as the plan validates it) and its translator ``starfinder.dataset.workflow._report_block``. Fixtures,
all written in the test: the contract's ``report`` block (W-383's ``markers.yaml`` shortened), ``report: {}``, an
unknown key and ``backend: matlab``. The report and compare commands that replaced the rules ``report_sample`` and
``report_dataset`` (W-750) are tested in test_workflow_report_tasks.py, the tasks the plan adds in
test_workflow_plan.py and test_workflow_declared_outputs.py.
"""
import copy
from dataclasses import fields
from pathlib import Path

import jsonschema
import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder.report import MarkerGroup, ReportConfig, ReportLimits, RuleLimit

from . import workflow_examples as X
from . import workflow_tasks as T
from .test_fov_membership import MATLAB_RSF

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
DEFS = SCHEMA["$defs"]
VALIDATOR = jsonschema.validators.validator_for(SCHEMA)(SCHEMA)
SAMPLE = "sample1"

# The block of design/specs/report.md, "Workflow configuration": W-383's markers.yaml for LN sample1, shortened.
REPORT = yaml.safe_load("""
markers:
  - name: lymph node compartments
    genes: [Cd3e, Cd79a, Lyz2, Pecam1]
  - name: T cell subsets
    genes: [Cd4, Cd8a, Foxp3]
marker_pairs:
  - [Cd3e, Cd79a]
  - [Cd4, Cd8a]
positive_threshold: 1
example_fovs: null              # or three FOV IDs
limits:
  shift_residual: {k: 5, floor: 5, absolute: null}
  alignment_score: {k: null, floor: null, absolute: null}
  correct_form_fraction: {k: 3, floor: 0.05, absolute: null}
  matched_fraction: {k: 3, floor: 0.05, absolute: null}
  spots_near_zero: 0.05
figure_width_px: 1600
jpeg_quality: 85
""")
REPORT_CONFIG = ReportConfig(
    markers=(MarkerGroup("lymph node compartments", ("Cd3e", "Cd79a", "Lyz2", "Pecam1")),
             MarkerGroup("T cell subsets", ("Cd4", "Cd8a", "Foxp3"))),
    marker_pairs=(("Cd3e", "Cd79a"), ("Cd4", "Cd8a")), positive_threshold=1, example_fovs=None,
    limits=ReportLimits(shift_residual=RuleLimit(5, 5), alignment_score=RuleLimit(),
                        correct_form_fraction=RuleLimit(3, 0.05), matched_fraction=RuleLimit(3, 0.05),
                        spots_near_zero=0.05),
    figure_width_px=1600, jpeg_quality=85)


def wf_config(name):
    return yaml.safe_load((ROOT / name).read_text())


def init_names(cls):
    return {f.name for f in fields(cls) if f.init}


def dataset_config(tmp_path, *, dataset_id="data", sample_id="sample", **blocks):
    """A Python configuration whose output root is tmp_path/<dataset_id>/output."""
    return {"backend": "python", "dataset_id": dataset_id, "sample_id": sample_id, "output_id": "output",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path),
            "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 64, "img_row": 48,
            "voxel_size_xy": 1.0, "voxel_size_z": 1.0, "rules": {}, **blocks}


def annotation(output_root, rows):
    """MATLAB's sample annotation (the MATLAB configurations)."""
    path = Path(output_root) / "documents" / "sample-annotation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample_id,fov_start,fov_end\n" + "".join(f"{s},{a},{b}\n" for s, a, b in rows))
    return path


def python_shape(tmp_path, **blocks):
    """A configuration in the Python shape (the worked example's dataset keys and processing block) with blocks."""
    return dict(X.dataset(tmp_path, "report"), processing=copy.deepcopy(X.PROCESSING), **copy.deepcopy(blocks))


# --- WF1: the schema equals the configuration ----------------------------------------------------------------

def test_wf1_report_keys_equal_the_config_fields():
    block = DEFS["report_block"]["properties"]
    assert set(block) == init_names(ReportConfig)
    assert set(block["markers"]["items"]["properties"]) == init_names(MarkerGroup)
    assert set(block["limits"]["properties"]) == init_names(ReportLimits)
    rules = {name for name, value in block["limits"]["properties"].items()
             if value == {"$ref": "#/$defs/report_rule_limit"}}
    assert rules == {f.name for f in fields(ReportLimits) if isinstance(f.default, RuleLimit)}
    assert set(DEFS["report_rule_limit"]["properties"]) == init_names(RuleLimit)
    assert SCHEMA["properties"]["report"] == {"$ref": "#/$defs/report_block"}
    for definition in (DEFS["report_block"], block["markers"]["items"], block["limits"], DEFS["report_rule_limit"]):
        assert definition["additionalProperties"] is False


def test_wf1_the_block_validates_with_the_python_backend(tmp_path):
    """The package validates the block (the Python schema and the block translator, as the plan does, W-750): the
    configuration without it, and with the contract's block, an empty block and the example FOVs."""
    config = python_shape(tmp_path)
    assert T.validation_error(config) is None
    for block in (REPORT, {}, {"example_fovs": ["F1", "F2", "F3"], "limits": {"shift_residual": {"absolute": 4}}}):
        assert T.validation_error(dict(config, report=copy.deepcopy(block))) is None


def test_wf1_the_block_is_python_only_in_the_schema():
    """The MATLAB workflow validates its configuration with the MATLAB schema before any rule runs
    (workflow/rules/legacy.smk), so a MATLAB run, which never reaches the adapter, must be refused there: backend
    matlab and an omitted backend (the schema's default, matlab) refuse the block, while the same configuration
    without it is valid. (The Python workflow validates with the package's schema: the test above.)"""
    base = wf_config("docs/examples/workflow-minimal.yaml")
    # The MATLAB backend names and selects its FOVs by fov_id_pattern, n_fovs and the subset keys (the Python one
    # by fovs.csv).
    without = {**{k: v for k, v in base.items() if k != "backend"}, "fov_id_pattern": "tile_{i}", "n_fovs": 1,
               "subset_list": [], "subset_range": False, "subset_start": 1, "subset_end": 1, "subset_random": False,
               "n_random_tests": 1, "rules": {"rsf_single_fov": MATLAB_RSF}}
    assert SCHEMA["properties"]["backend"]["default"] == "matlab"
    for block in (REPORT, {}):
        VALIDATOR.validate(dict(without, backend="matlab"))
        VALIDATOR.validate(without)
        with pytest.raises(jsonschema.ValidationError):
            VALIDATOR.validate(dict(without, backend="matlab", report=copy.deepcopy(block)))
        with pytest.raises(jsonschema.ValidationError):
            VALIDATOR.validate(dict(without, report=copy.deepcopy(block)))


@pytest.mark.parametrize("change", [
    {"title": "LN"},                                                         # an unknown key
    {"markers": [{"name": "too many", "genes": ["A", "B", "C", "D", "E"]}]},  # five genes
    {"markers": [{"name": "x", "genes": ["A"], "colour": "red"}]},
    {"marker_pairs": [["Cd3e"]]},
    {"marker_pairs": [["Cd3e", "Cd3e"]]},
    {"positive_threshold": 0},
    {"example_fovs": ["F1", "F2"]},
    {"limits": {"shift_residual": {"k": -1}}},
    {"limits": {"shift_residual": {"sigma": 3}}},
    {"limits": {"spots": 0.1}},
    {"limits": {"spots_near_zero": 1}},
    {"figure_width_px": 0},
    {"jpeg_quality": 96},
])
def test_wf1_invalid_blocks_fail_the_schema(tmp_path, change):
    """The package refuses each invalid block (the Python schema or the block translator, as the plan validates it;
    W-750)."""
    assert T.validation_error(python_shape(tmp_path, report=dict(copy.deepcopy(REPORT), **change)))


# --- WF1: the adapter ----------------------------------------------------------------------------------------

def test_wf1_the_contract_block_translates(tmp_path):
    config = dataset_config(tmp_path, report=copy.deepcopy(REPORT))
    assert workflow._report_block(config) == REPORT_CONFIG
    changed = dict(copy.deepcopy(REPORT), example_fovs=["F3", "F1", "F2"], positive_threshold=2,
                   limits={"alignment_score": {"absolute": 0.4}, "spots_near_zero": 0.1})
    translated = workflow._report_block(dataset_config(tmp_path, report=changed))
    assert translated.example_fovs == ("F3", "F1", "F2") and translated.positive_threshold == 2
    # An omitted rule keeps its default; an omitted key of a given rule is null.
    assert translated.limits == ReportLimits(alignment_score=RuleLimit(absolute=0.4), spots_near_zero=0.1)
    assert all(isinstance(pair, tuple) for pair in translated.marker_pairs)


@pytest.mark.parametrize("block", [{}, None])
def test_wf1_an_empty_block_gives_the_defaults(tmp_path, block):
    assert workflow._report_block(dataset_config(tmp_path, report=block)) == ReportConfig()


@pytest.mark.parametrize("block, match", [
    ({"title": "LN"}, r"unknown report keys: \['title'\]"),
    ({"markers": [{"name": "x", "genes": ["A"], "colour": "red"}]}, r"unknown report\.markers keys: \['colour'\]"),
    ({"limits": {"spots": 0.1}}, r"unknown report\.limits keys: \['spots'\]"),
    ({"limits": {"shift_residual": {"sigma": 3}}}, r"unknown report\.limits\.shift_residual keys: \['sigma'\]"),
])
def test_wf1_an_unknown_key_raises_naming_it(tmp_path, block, match):
    with pytest.raises(ValueError, match=match):
        workflow._report_block(dataset_config(tmp_path, report=block))


@pytest.mark.parametrize("backend", ["matlab", None])
def test_wf1_the_block_needs_the_python_backend(tmp_path, backend):
    config = dataset_config(tmp_path, report=copy.deepcopy(REPORT))
    if backend is None:
        del config["backend"]
    else:
        config["backend"] = backend
    with pytest.raises(ValueError, match=r"report is Python-only and need backend: python"):
        workflow._report_block(config)
