"""WF1 of docs/report-algorithms.md: the workflow's ``report`` block and the rules ``report_sample`` and
``report_dataset`` (W-458; docs/report-contract.md, "Workflow configuration", option WR1).

The Python-only ``report`` block of ``workflow/schemas/config.schema.yaml``, its adapter
``starfinder.dataset.workflow._report_block``, the rules of ``workflow/rules/report.smk`` and
their scripts. Fixtures, all written in the test: the repository's workflow configurations
(``docs/examples/workflow-*.yaml`` and ``tests/*.yaml``) as they are and with the contract's
``report`` block (W-383's ``markers.yaml`` shortened), ``report: {}``, an unknown key and
``backend: matlab``. The scripts run through a stub ``snakemake`` object, as in
``test_workflow_scripts.py``, with ``Dataset.report`` and ``Dataset.compare_samples`` spied;
one case runs the real entries on ``report_records`` of ``report_fixtures.py``. The rule
set and the requested outputs are read from the rule files and from ``get_overall_output``
of ``common.smk``, executed with the names it defines; Snakemake itself never runs.
"""
import copy
import re
import runpy
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pandas as pd
import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder import _snakemake
from starfinder.dataset import Dataset
from starfinder.report import MarkerGroup, ReportConfig, ReportLimits, RuleLimit, load_report

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
DEFS = SCHEMA["$defs"]
VALIDATOR = jsonschema.validators.validator_for(SCHEMA)(SCHEMA)
WF_CONFIGS = ("docs/examples/workflow-full.yaml", "docs/examples/workflow-minimal.yaml",
              "docs/examples/workflow-recipe-2.yaml", "tests/minimal_config.yaml", "tests/tissue_2D_test.yaml")
SAMPLE = "sample1"

# The block of docs/report-contract.md, "Workflow configuration": W-383's markers.yaml for LN sample1, shortened.
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
MULTIVIEW = {"method": "multiview_stitcher", "positions": {"source": "regular_grid",
                                                           "regular_grid": {"columns": 2, "rows": 2,
                                                                            "order": "column_major_down_right",
                                                                            "overlap": 0.25}}}


class Wildcards(dict):
    """Snakemake wildcards: read both as attributes and by key."""
    __getattr__ = dict.__getitem__


def wf_config(name):
    return yaml.safe_load((ROOT / name).read_text())


def init_names(cls):
    return {f.name for f in fields(cls) if f.init}


def dataset_config(tmp_path, *, dataset_id="data", sample_id="sample", pattern="Position{i:03d}", **blocks):
    """A Python configuration whose output root is tmp_path/<dataset_id>/output."""
    return {"backend": "python", "dataset_id": dataset_id, "sample_id": sample_id, "output_id": "output",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path), "fov_id_pattern": pattern,
            "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 64, "img_row": 48,
            "voxel_size_xy": 1.0, "voxel_size_z": 1.0, "rules": {}, **blocks}


def annotation(output_root, rows):
    path = Path(output_root) / "documents" / "sample-annotation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample_id,fov_start,fov_end\n" + "".join(f"{s},{a},{b}\n" for s, a, b in rows))
    return path


def run_script(name, snakemake):
    runpy.run_path(str(ROOT / "workflow" / "scripts" / name), init_globals={"snakemake": snakemake})


def rule_names(text):
    return re.findall(r"^\s*rule (\w+):", text, flags=re.MULTILINE)


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


@pytest.mark.parametrize("name", WF_CONFIGS[:3])
def test_wf1_the_block_validates_with_the_python_backend(name):
    config = wf_config(name)
    assert config["backend"] == "python" and "report" not in config
    VALIDATOR.validate(config)
    for block in (REPORT, {}, {"example_fovs": ["F1", "F2", "F3"], "limits": {"shift_residual": {"absolute": 4}}}):
        VALIDATOR.validate(dict(config, report=copy.deepcopy(block)))


def test_wf1_the_block_is_python_only_in_the_schema():
    """The workflow validates its configuration with the schema before any rule runs (workflow/Snakefile), so a
    MATLAB run, which never reaches the adapter, must be refused there: backend matlab and an omitted backend
    (the schema's default, matlab) refuse the block, while the same configuration without it is valid."""
    base = wf_config("docs/examples/workflow-minimal.yaml")
    without = {k: v for k, v in base.items() if k != "backend"}
    assert SCHEMA["properties"]["backend"]["default"] == "matlab"
    for block in (REPORT, {}):
        VALIDATOR.validate(dict(base, report=copy.deepcopy(block)))
        VALIDATOR.validate(dict(base, backend="matlab"))
        VALIDATOR.validate(without)
        with pytest.raises(jsonschema.ValidationError):
            VALIDATOR.validate(dict(base, backend="matlab", report=copy.deepcopy(block)))
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
def test_wf1_invalid_blocks_fail_the_schema(change):
    config = dict(wf_config("docs/examples/workflow-minimal.yaml"), report=dict(copy.deepcopy(REPORT), **change))
    with pytest.raises(jsonschema.ValidationError):
        VALIDATOR.validate(config)


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


def test_wf1_the_block_is_checked_when_the_rule_files_are_parsed():
    common = (ROOT / "workflow/rules/common.smk").read_text()
    assert 'REPORT_BLOCK = "report" in config' in common
    python = common[common.index('if BACKEND == "python":\n    sys.path.insert'):]
    assert re.search(r"\n    if REPORT_BLOCK:\n        _report_block\(config\)\n", python)


# --- WF1: the rules (WR1) ------------------------------------------------------------------------------------

def test_wf1_report_sample_declares_the_last_per_sample_output(tmp_path):
    root, fovs = Path("/out"), ["Position001", "Position002"]
    assembly = dataset_config(tmp_path, stitching=MULTIVIEW, assembly={"assignment": "default"}, report={})
    stitching = dataset_config(tmp_path, stitching=MULTIVIEW, report={})
    processing = dataset_config(tmp_path, report={})
    assert _snakemake._report_sample_files(assembly, root, fovs, SAMPLE) == (
        ["/out/export/sample1/export.json"], "/out/report/sample1/report.json")
    # An export block configures export_sample, which an assembly block already runs: the same input.
    exported = dict(assembly, export={"route": "per_fov"})
    assert _snakemake._report_sample_files(exported, root, fovs, SAMPLE)[0] == ["/out/export/sample1/export.json"]
    assert _snakemake._report_sample_files(stitching, root, fovs, SAMPLE)[0] == ["/out/stitching/sample1/placement.json"]
    assert _snakemake._report_sample_files(processing, root, fovs, SAMPLE)[0] == [
        "/out/fovs/Position001/run.json", "/out/fovs/Position002/run.json"]
    for config in (assembly, stitching, processing):
        assert _snakemake._report_sample_files(config, root, [])[1] == "/out/report/{sample}/report.json"
    assert _snakemake._report_dataset_files(root, ["s2", "s1"]) == (
        ["/out/report/s2/report.json", "/out/report/s1/report.json"], "/out/report/samples.csv")


def test_wf1_report_smk_declares_exactly_these():
    text = (ROOT / "workflow/rules/report.smk").read_text()
    assert rule_names(text) == ["report_sample", "report_dataset"]
    assert re.search(r"def get_report_input\(wildcards\):\n(?:    .*\n)+?    return _report_sample_files\(config, "
                     r"OUTPUT_DIR, SAMPLE_TO_FOV\[wildcards\.sample\], wildcards\.sample\)\[0\]", text)
    sample_rule = text[text.index("rule report_sample:"):text.index("rule report_dataset:")]
    assert re.search(r"input:\s+get_report_input\s+output:\s+_report_sample_files\(config, OUTPUT_DIR, \[\]\)\[1\]",
                     sample_rule)
    assert 'script:\n        "../scripts/report_sample.py"' in sample_rule
    dataset_rule = text[text.index("rule report_dataset:"):]
    assert re.search(r"input:\s+_report_dataset_files\(OUTPUT_DIR, REPORT_SAMPLES\)\[0\]\s+output:\s+"
                     r"_report_dataset_files\(OUTPUT_DIR, REPORT_SAMPLES\)\[1\]", dataset_rule)
    assert 'script:\n        "../scripts/report_dataset.py"' in dataset_rule
    assert "REPORT_SAMPLES = [sample for sample in SAMPLE_ANNOT['sample_id'].unique() if sample in SAMPLE]" in text


def spy(monkeypatch, name):
    calls = []
    monkeypatch.setattr(Dataset, name, lambda self, *args, **kwargs: calls.append((self, args, kwargs)))
    return calls


@pytest.mark.parametrize("blocks, assignment", [
    ({"stitching": MULTIVIEW, "assembly": {"assignment": "default"}}, "default"),
    ({"stitching": MULTIVIEW, "assembly": {"assignment": "cells_v2"}}, "cells_v2"),
    ({"stitching": MULTIVIEW, "assembly": {"assignment": None}}, None),
    ({"stitching": MULTIVIEW, "assembly": {}}, "default"),
    ({"stitching": MULTIVIEW}, None),
    ({}, None),
])
@pytest.mark.parametrize("block", ["contract", "empty"])
def test_wf1_report_sample_calls_dataset_report(tmp_path, monkeypatch, blocks, assignment, block):
    """The sample's FOVs from the annotation, the assembly block's assignment (none without the block, where no
    rule saves an assignment) and the report block's configuration."""
    calls = spy(monkeypatch, "report")
    report = copy.deepcopy(REPORT) if block == "contract" else {}
    config = dataset_config(tmp_path, report=report, **copy.deepcopy(blocks))
    output_root = tmp_path / "data" / "output"
    annotation(output_root, [("sample0", 1, 2), (SAMPLE, 3, 5), ("sample2", 6, 6)])
    inputs, output = _snakemake._report_sample_files(config, output_root, ["Position003", "Position004", "Position005"],
                                                   SAMPLE)
    run_script("report_sample.py", SimpleNamespace(config=config, input=inputs, output=[output],
                                                   wildcards=Wildcards(sample=SAMPLE)))
    (dataset, args, kwargs), = calls
    assert dataset.output_root == output_root and dataset.dataset_id == "data"
    assert args == (SAMPLE, ["Position003", "Position004", "Position005"])
    assert kwargs == {"assignment": assignment, "config": REPORT_CONFIG if block == "contract" else ReportConfig(),
                      "output_dir": output_root / "report" / SAMPLE}


def test_wf1_report_dataset_calls_compare_samples_on_every_sample(tmp_path, monkeypatch):
    calls = spy(monkeypatch, "compare_samples")
    config = dataset_config(tmp_path, report={})
    output_root = tmp_path / "data" / "output"
    inputs, output = _snakemake._report_dataset_files(output_root, ["sample0", SAMPLE, "sample2"])
    run_script("report_dataset.py", SimpleNamespace(config=config, input=inputs, output=[output],
                                                    wildcards=Wildcards()))
    (dataset, args, kwargs), = calls
    assert dataset.output_root == output_root
    assert args == (["sample0", SAMPLE, "sample2"],) and kwargs == {"report_root": output_root / "report"}


def test_wf1_the_rules_raise_before_calling_the_entries(tmp_path, monkeypatch):
    calls = (spy(monkeypatch, "report"), spy(monkeypatch, "compare_samples"))
    output_root = tmp_path / "data" / "output"
    annotation(output_root, [(SAMPLE, 1, 2)])
    for script, bad in (("report_sample.py", {"title": "LN"}), ("report_dataset.py", {"title": "LN"})):
        config = dataset_config(tmp_path, report=bad)
        stub = SimpleNamespace(config=config, input=[str(output_root / "report" / SAMPLE / "report.json")],
                               output=[str(output_root / "report" / SAMPLE / "report.json")],
                               wildcards=Wildcards(sample=SAMPLE))
        with pytest.raises(ValueError, match=r"unknown report keys: \['title'\]"):
            run_script(script, stub)
    assert calls == ([], [])


def test_wf1_the_rules_write_the_report_and_the_comparison_page(tmp_path):
    """The real entries through both scripts on report_records (hand-written run records and placement of sample
    s1, FOVs F1 to F4): report_sample writes report/s1/ with its report.json, report_dataset report/samples.csv."""
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records
    from .report_fixtures import SAMPLE as REPORT_SAMPLE
    config = dataset_config(tmp_path, dataset_id=DATASET, sample_id=SAMPLE_ID, pattern="F{i}",
                            stitching=MULTIVIEW, report={"marker_pairs": [["G1", "G2"]]})
    output_root = tmp_path / DATASET / "output"
    case = report_records(output_root)
    assert list(case.fov_ids) == list(FOVS)
    annotation(output_root, [(REPORT_SAMPLE, 1, 4)])
    inputs, output = _snakemake._report_sample_files(config, output_root, list(FOVS), REPORT_SAMPLE)
    assert inputs == [str(output_root / "stitching" / REPORT_SAMPLE / "placement.json")] and Path(inputs[0]).is_file()
    run_script("report_sample.py", SimpleNamespace(config=config, input=inputs, output=[output],
                                                   wildcards=Wildcards(sample=REPORT_SAMPLE)))
    record = load_report(Path(output).parent)
    assert Path(output).is_file() and (Path(output).parent / "report.html").is_file()
    assert record["sample"] == REPORT_SAMPLE and record["fov_ids"] == list(FOVS)
    inputs, output = _snakemake._report_dataset_files(output_root, [REPORT_SAMPLE])
    run_script("report_dataset.py", SimpleNamespace(config=config, input=inputs, output=[output],
                                                    wildcards=Wildcards()))
    samples = pd.read_csv(output)
    assert samples.iloc[:, 0].tolist() == [REPORT_SAMPLE] and "fovs_expected" in samples
    assert (output_root / "report" / "index.html").is_file()


# --- WF1: configurations without the block -------------------------------------------------------------------

def common_preamble(config, output_root, monkeypatch):
    """The names common.smk defines before its first rule, executed with ``config`` (its sample annotation is
    written under ``output_root`` first)."""
    text = (ROOT / "workflow/rules/common.smk").read_text()
    source = text[:text.index("### ==================== [ Preparation Rule ]")]
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"config": config, "workflow": SimpleNamespace(basedir=str(ROOT / "workflow"))}
    exec(compile(source, "common.smk", "exec"), namespace)
    assert namespace["OUTPUT_DIR"] == Path(output_root)
    return namespace


def repository_config(name, tmp_path):
    config = wf_config(name)
    config["root_output_path"] = str(tmp_path)
    for key in ("subset_list", "subset_range", "subset_random"):
        config.setdefault(key, False if key != "subset_list" else [])
    output_root = tmp_path / config["dataset_id"] / config["output_id"]
    annotation(output_root, [(SAMPLE, 1, config["n_fovs"]), ("sample2", config["n_fovs"] + 1, config["n_fovs"] + 1)])
    config["n_fovs"] += 1
    config["subset_range"] = False
    return config, output_root


@pytest.mark.parametrize("name", WF_CONFIGS)
def test_wf1_without_the_block_no_rule_and_no_output_is_added(tmp_path, monkeypatch, name):
    config, output_root = repository_config(name, tmp_path)
    assert "report" not in config
    names = common_preamble(config, output_root, monkeypatch)
    assert names["REPORT_BLOCK"] is False
    requested = names["get_overall_output"](None)
    assert not any("/report/" in path for path in requested)
    # The same configuration with the block requests exactly the reports in addition, in that order.
    names["REPORT_BLOCK"] = True
    with_block = names["get_overall_output"](None)
    reports = [f"{output_root}/report/{sample}/report.json" for sample in names["SAMPLE"]]
    assert [p for p in with_block if p not in requested] == reports + [f"{output_root}/report/samples.csv"]
    assert [p for p in with_block if "/report/" not in p] == requested
    # The Snakefile includes report.smk with the block only, and no other rule file names a report rule.
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    assert re.search(r'\nif REPORT_BLOCK:\n    include: "rules/report.smk"\n', snakefile)
    assert snakefile.count('include: "rules/report.smk"') == 1
    for path in (ROOT / "workflow/rules").glob("*.smk"):
        if path.name != "report.smk":
            assert not {"report_sample", "report_dataset"} & set(rule_names(path.read_text())), path.name


def test_wf1_with_the_block_the_report_rules_are_requested(tmp_path, monkeypatch):
    config, output_root = repository_config("docs/examples/workflow-minimal.yaml", tmp_path)
    names = common_preamble(dict(config, report={}), output_root, monkeypatch)
    assert names["REPORT_BLOCK"] is True and names["SAMPLE"] == [SAMPLE, "sample2"]
    requested = names["get_overall_output"](None)
    assert [p for p in requested if "/report/" in p] == [f"{output_root}/report/{SAMPLE}/report.json",
                                                         f"{output_root}/report/sample2/report.json",
                                                         f"{output_root}/report/samples.csv"]
