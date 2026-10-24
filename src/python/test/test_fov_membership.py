"""The Python workflow's FOV membership (W-566): ``<input_root>/fovs.csv`` and ``selected_fovs``.

``starfinder.dataset.workflow._membership`` reads ``fovs.csv`` (``sample,fov_id`` and an optional
``position_id``) in row order and restricts it by ``selected_fovs``; ``common.smk`` builds ``FOVS``,
``SAMPLE`` and ``SAMPLE_TO_FOV`` from it for the Python backend and keeps the sample annotation, the FOV
pattern and the subset keys for the MATLAB backend; the per-sample adapters read the same restricted
membership. Fixtures, all written in the test: the repository's example configurations with a
``fovs.csv`` (or, for the MATLAB backend, a sample annotation) under ``tmp_path``. The rule files are
executed in part, as in ``test_report_workflow.py`` (the names ``common.smk`` defines before its first
rule, its Python checks after the rules, and the helpers of ``stitching-py.smk`` before its rules);
Snakemake itself never runs.
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder import _snakemake
from starfinder.dataset import Dataset

from .stitching_fixtures import maf_text

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_merged",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                            "overlap": 0.25}},
             "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0}}
# Two samples whose rows interleave, so that row order, not sorting, decides both orders.
ROWS = [("s2", "F10", 7), ("s1", "F2", 2), ("s2", "F3", 3), ("s1", "F1", 1)]


class Wildcards(dict):
    """Snakemake wildcards: read both as attributes and by key."""
    __getattr__ = dict.__getitem__


def config_for(tmp_path, **extra):
    """A Python configuration whose input root is tmp_path/input/data/sample and output root tmp_path/data/output."""
    return {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path),
            "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 64, "img_row": 48,
            "voxel_size_xy": 1.0, "voxel_size_z": 1.0, "rules": {}, **extra}


def write_fovs(config, text):
    path = Path(config["root_input_path"]) / config["dataset_id"] / config["sample_id"] / "fovs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def rows_text(rows=ROWS, positions=True):
    header = "sample,fov_id,position_id\n" if positions else "sample,fov_id\n"
    return header + "".join(f"{s},{f},{p}\n" if positions else f"{s},{f}\n" for s, f, p in rows)


# --- fovs.csv: row order, position_id and selected_fovs --------------------------------------------------------

def test_membership_is_fovs_csv_in_row_order(tmp_path):
    config = config_for(tmp_path)
    path = write_fovs(config, rows_text())
    members = workflow._membership(config)
    assert members.samples == {"s2": ("F10", "F3"), "s1": ("F2", "F1")}
    assert members.fovs == ("F10", "F2", "F3", "F1")
    assert members.position_ids == {"F10": 7, "F2": 2, "F3": 3, "F1": 1}
    assert workflow._membership(config, path) == members
    assert workflow._sample_fovs(config, "s1") == ["F2", "F1"]
    write_fovs(config, rows_text(positions=False))
    assert workflow._membership(config).position_ids is None


def test_selected_fovs_restricts_the_membership(tmp_path):
    config = config_for(tmp_path, selected_fovs=["F1", "F10", "F3"])
    write_fovs(config, rows_text())
    members = workflow._membership(config)
    # fovs.csv's order, not the list's; a sample without a selected FOV is left out.
    assert members.samples == {"s2": ("F10", "F3"), "s1": ("F1",)}
    assert members.position_ids == {"F10": 7, "F3": 3, "F1": 1}
    assert workflow._membership(dict(config, selected_fovs=["F2"])).samples == {"s1": ("F2",)}
    with pytest.raises(ValueError, match=r"sample 's2' has no FOV of the run"):
        workflow._sample_fovs(dict(config, selected_fovs=["F2"]), "s2")
    assert workflow._membership(dict(config, selected_fovs=None)).fovs == ("F10", "F2", "F3", "F1")


@pytest.mark.parametrize("text, match", [
    (rows_text() + "s1,F1,1\n", r"repeats the rows of the FOVs \['F1'\]"),
    (rows_text() + "s2,F1,1\n", r"gives the FOVs \['F1'\] conflicting rows"),            # another sample
    (rows_text() + "s1,F1,9\n", r"gives the FOVs \['F1'\] conflicting rows"),            # another position_id
    (rows_text() + "s1,F4,3\n", r"gives the position_id \[3\] to more than one FOV"),
    ("sample,fov_id,position_id\ns1,F1,one\n", r"position_id must be an integer PositionID; got \['one'\]"),
    ("sample,fov_id\ns1,\n", r"rows \[2\] do not give a value"),
    ("sample,fov_id\ns1,F1,3\n", r"rows \[2\] do not give a value"),
    ("sample_id,fov_start,fov_end\ns1,1,2\n", r"must have the columns sample,fov_id and optionally position_id"),
    ("sample,fov_id,condition\ns1,F1,a\n", r"must have the columns sample,fov_id and optionally position_id"),
    ("sample,fov_id\n", r"lists no FOV"),
])
def test_repeated_conflicting_and_malformed_rows_raise(tmp_path, text, match):
    config = config_for(tmp_path)
    write_fovs(config, text)
    with pytest.raises(ValueError, match=match):
        workflow._membership(config)


@pytest.mark.parametrize("selected, match", [
    (["F1", "F9"], r"unknown \['F9'\], repeated \[\]"),
    (["F1", "F1"], r"unknown \[\], repeated \['F1'\]"),
    ([], "nonempty list"),
    ("F1", "nonempty list"),
])
def test_bad_selected_fovs_raise(tmp_path, selected, match):
    config = config_for(tmp_path, selected_fovs=selected)
    write_fovs(config, rows_text())
    with pytest.raises(ValueError, match=match):
        workflow._membership(config)


def test_a_missing_membership_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fovs.csv"):
        workflow._membership(config_for(tmp_path))


def test_the_schema_refuses_the_matlab_fov_keys_for_python_and_requires_them_for_matlab():
    import jsonschema
    schema = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
    validator = jsonschema.validators.validator_for(schema)(schema)
    config = yaml.safe_load((ROOT / "docs/examples/workflow-minimal.yaml").read_text())
    validator.validate(dict(config, selected_fovs=["tile_1"]))
    matlab = dict(config, backend="matlab", fov_id_pattern="tile_{i}", n_fovs=1, subset_list=[], subset_range=False,
                  subset_start=1, subset_end=1, subset_random=False, n_random_tests=1)
    validator.validate(matlab)
    for key, value in (("fov_id_pattern", "tile_{i}"), ("n_fovs", 1), ("subset_list", []), ("subset_range", False),
                       ("subset_start", 1), ("subset_end", 1), ("subset_random", False), ("n_random_tests", 1)):
        assert not validator.is_valid(dict(config, **{key: value})), key
    for invalid in ({k: v for k, v in matlab.items() if k != "n_fovs"}, dict(matlab, selected_fovs=["tile_1"]),
                    dict(config, selected_fovs=[]), dict(config, selected_fovs=["tile_1", "tile_1"])):
        assert not validator.is_valid(invalid)


# --- common.smk: FOVS, SAMPLE and SAMPLE_TO_FOV by backend -------------------------------------------------------

def common_names(config, monkeypatch, *, checks=False):
    """The names common.smk defines before its first rule; with ``checks`` also its Python checks after the rules."""
    text = (ROOT / "workflow/rules/common.smk").read_text()
    source = text[:text.index("### ==================== [ Preparation Rule ]")]
    if checks:
        source += text[text.index("# The FOV molecule table that reads_assignment"):]
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"config": config, "workflow": SimpleNamespace(basedir=str(ROOT / "workflow"))}
    exec(compile(source, "common.smk", "exec"), namespace)
    return namespace


def python_example(tmp_path, **extra):
    config = yaml.safe_load((ROOT / "docs/examples/workflow-minimal.yaml").read_text())
    config.update(root_input_path=str(tmp_path / "input"), root_output_path=str(tmp_path), **extra)
    return config


def test_common_smk_builds_the_python_membership_from_fovs_csv(tmp_path, monkeypatch):
    config = python_example(tmp_path)
    write_fovs(config, rows_text(positions=False))
    assert not (tmp_path / "example" / "direct" / "documents").exists()
    names = common_names(config, monkeypatch)
    assert names["FOVS"] == ["F10", "F2", "F3", "F1"]
    assert names["SAMPLE"] == ["s2", "s1"]
    assert names["SAMPLE_TO_FOV"] == {"s2": ["F10", "F3"], "s1": ["F2", "F1"]}
    assert names["FOV_TO_SAMPLE"] == {"F10": "s2", "F3": "s2", "F2": "s1", "F1": "s1"}
    assert names["MEMBERSHIP"] == str(tmp_path / "input" / "example" / "sample" / "fovs.csv")
    assert "SAMPLE_ANNOT" not in names
    # The same file gives the same values, in the same order, on every parse.
    again = common_names(copy.deepcopy(config), monkeypatch)
    assert [again[k] for k in ("FOVS", "SAMPLE", "SAMPLE_TO_FOV")] == [names[k] for k in ("FOVS", "SAMPLE",
                                                                                         "SAMPLE_TO_FOV")]


def matlab_config(tmp_path, **extra):
    config = yaml.safe_load((ROOT / "tests/minimal_config.yaml").read_text())
    assert config.get("backend", "matlab") == "matlab"
    config.update({"root_output_path": str(tmp_path), "fov_id_pattern": "Position{i:03d}", "n_fovs": 5,
                   "subset_list": [], "subset_range": False, "subset_random": False, **extra})
    annotation = tmp_path / config["dataset_id"] / config["output_id"] / "documents" / "sample-annotation.csv"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text("sample_id,fov_start,fov_end\nsB,4,5\nsA,1,3\n")
    return config


def base_rule(config):
    """common.smk's MATLAB rule as it stands on the base revision: the annotation's ranges by fov_id_pattern, FOVS
    the sorted names of 1..n_fovs, then the subset keys."""
    pattern = config["fov_id_pattern"].format
    to_fov = {"sB": [pattern(i=i) for i in (4, 5)], "sA": [pattern(i=i) for i in (1, 2, 3)]}
    fovs = sorted(pattern(i=j) for j in range(1, config["n_fovs"] + 1))
    samples = list(to_fov)
    if config["subset_list"]:
        fovs = [pattern(i=j) for j in config["subset_list"]]
        samples = {s for s, members in to_fov.items() for f in fovs if f in members}
    elif config["subset_range"]:
        fovs = [pattern(i=j) for j in range(config["subset_start"], config["subset_end"] + 1)]
    return fovs, samples, to_fov


@pytest.mark.parametrize("subset", [{}, {"subset_list": [5, 2]}, {"subset_range": True, "subset_start": 2,
                                                                     "subset_end": 4}])
def test_common_smk_keeps_the_matlab_membership(tmp_path, monkeypatch, subset):
    """The MATLAB backend reads its sample annotation, fov_id_pattern, n_fovs and the subset keys as on the base
    (subset_list's SAMPLE is a set there, so its order is not compared)."""
    config = matlab_config(tmp_path, **subset)
    names = common_names(config, monkeypatch)
    fovs, samples, to_fov = base_rule(config)
    assert names["FOVS"] == fovs and names["SAMPLE_TO_FOV"] == to_fov
    assert (set(names["SAMPLE"]) == samples) if subset.get("subset_list") else (names["SAMPLE"] == samples)
    assert names["MEMBERSHIP"].endswith("documents/sample-annotation.csv")


def test_common_smk_keeps_the_matlab_random_subset(tmp_path, monkeypatch):
    config = matlab_config(tmp_path, subset_random=True, n_random_tests=2)
    names = common_names(config, monkeypatch)
    assert len(names["FOVS"]) == 2 and set(names["FOVS"]) <= set(base_rule(config)[0])


def test_selected_fovs_restricts_every_requested_output(tmp_path, monkeypatch):
    """With selected_fovs, the per-FOV outputs and the per-sample ones (stitching, export, report) are requested
    for the selected FOVs and their samples only."""
    config = python_example(tmp_path, selected_fovs=["F3", "F10"], stitching=MULTIVIEW,
                            assembly={"assignment": None}, report={})
    write_fovs(config, rows_text(positions=False))
    names = common_names(config, monkeypatch, checks=True)
    assert names["FOVS"] == ["F10", "F3"] and names["SAMPLE"] == ["s2"]
    assert names["SAMPLE_TO_FOV"] == {"s2": ["F10", "F3"]}
    output = Path(tmp_path / "example" / "direct")
    requested = names["get_overall_output"](None)
    assert [p for p in requested if p.endswith("run.json")] == [f"{output}/fovs/F10/run.json",
                                                                 f"{output}/fovs/F3/run.json"]
    assert [p for p in requested if "/stitching/" in p or "/export/" in p or "/report/" in p] == [
        f"{output}/stitching/s2/placement.json", f"{output}/export/s2/export.json",
        f"{output}/report/s2/report.json", f"{output}/report/samples.csv"]
    # The export rule's input function (export.smk) gives the selected FOVs of the sample.
    inputs = _snakemake._export_sample_files(config, output, names["SAMPLE_TO_FOV"]["s2"], "s2")[0]
    assert inputs[1:] == [f"{output}/fovs/F10/molecules.csv", f"{output}/fovs/F3/molecules.csv"]


def test_a_per_sample_rule_reads_the_selected_fovs_only(tmp_path, monkeypatch):
    """stitch_sample and report_sample, the per-sample adapters, read the sample's selected FOVs; reads_assignment
    takes its sample and tile-configuration id from the same membership."""
    calls = []
    monkeypatch.setattr(Dataset, "stitch", lambda self, *args, **kwargs: calls.append(("stitch", args, kwargs)))
    monkeypatch.setattr(Dataset, "report", lambda self, *args, **kwargs: calls.append(("report", args, kwargs)))
    config = config_for(tmp_path, selected_fovs=["F1", "F10"], stitching=MULTIVIEW, report={})
    path = write_fovs(config, rows_text())
    placement = tmp_path / "data/output/stitching/s1/placement.json"
    _snakemake._run_stitching(SimpleNamespace(config=config, input=[str(path)], output=[str(placement)],
                                              wildcards=Wildcards(sample="s1")))
    report = tmp_path / "data/output/report/s1/report.json"
    _snakemake._run_report_sample(SimpleNamespace(config=config, input=[], output=[str(report)],
                                                  wildcards=Wildcards(sample="s1")))
    (_, stitch_args, stitch_kwargs), (_, report_args, _) = calls
    assert stitch_args == ("s1", ["F1"]) and report_args == ("s1", ["F1"])
    assert stitch_kwargs["position_ids"] is None       # a regular grid reads no PositionID
    assert _snakemake._fov_sample(path, config, "F10") == ("s2", 7)
    assert _snakemake._fov_sample(path, config, "F1") == ("s1", 1)
    with pytest.raises(ValueError, match="F2 is in no sample"):
        _snakemake._fov_sample(path, config, "F2")
    write_fovs(config, rows_text(positions=False))
    # Without position_id the id is the FOV's 1-based place among its sample's FOVs of the run.
    assert _snakemake._fov_sample(path, dict(config, selected_fovs=None), "F3") == ("s2", 2)
    assert _snakemake._fov_sample(path, config, "F10") == ("s2", 1)


# --- The legacy sample rules and the stitching parse for the Python backend ------------------------------------

@pytest.mark.parametrize("rules, blocks, match", [
    (["stitching_preparation", "create_tile_config"], {}, r"\['stitching_preparation', 'create_tile_config'\] need a "
                                                         r"stitching block"),
    (["create_sample_h5ad"], {"stitching": MULTIVIEW}, r"\['create_sample_h5ad'\] need a stitching block and an "
                                                        r"assembly block"),
    (["create_sample_reads_assignment"], {}, "need a stitching block and an assembly block"),
    (["reads_assignment"], {}, r"\['reads_assignment'\] needs an assembly block, or a stitching block with method "
                               r"bigstitcher"),
])
def test_the_legacy_sample_rules_are_refused_for_python_naming_the_blocks(tmp_path, monkeypatch, rules, blocks, match):
    config = python_example(tmp_path, **blocks)
    config["rules"].update({rule: {"run": True} for rule in rules})
    write_fovs(config, rows_text(positions=False))
    with pytest.raises(ValueError, match=r"the legacy sample rules run with backend matlab only; .*" + match):
        common_names(config, monkeypatch, checks=True)


def test_with_the_blocks_the_legacy_sample_rules_are_left_out(tmp_path, monkeypatch):
    config = python_example(tmp_path, stitching=MULTIVIEW, assembly={"assignment": None})
    config["rules"].update({rule: {"run": True} for rule in workflow._LEGACY_STITCHING_RULES
                            + workflow._LEGACY_SAMPLE_RULES})
    write_fovs(config, rows_text(positions=False))
    requested = common_names(config, monkeypatch, checks=True)["get_overall_output"](None)
    assert not any("/images/fused/" in p or "tile_config" in p or "_raw.h5ad" in p or "_reads_assignment" in p
                   for p in requested)


def test_the_snakefile_includes_the_legacy_sample_rules_for_matlab_only():
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    assert 'elif BACKEND == "matlab":\n    include: "rules/stitching.smk"' in snakefile
    assert 'if BACKEND == "matlab":\n    include: "rules/utils.smk"' in snakefile
    assert 'if BACKEND == "matlab":\n    localrules: create_sample_maf' in snakefile
    reads = (ROOT / "workflow/rules/reads-assignment.smk").read_text()
    legacy = reads[reads.index('if BACKEND == "matlab":'):]
    assert "    rule create_sample_h5ad:" in legacy and "    rule create_sample_reads_assignment:" in legacy
    assert "rule reads_assignment:" not in legacy
    assert "rule create_sample_maf:" in (ROOT / "workflow/rules/utils.smk").read_text()


def test_the_python_stitching_parse_needs_only_fovs_csv_and_a_stage_file(tmp_path, monkeypatch):
    """A Python configuration with a stitching block on a whole-acquisition stage file: common.smk and the helpers of
    stitching-py.smk parse with fovs.csv and the stage file alone (no sample annotation, no documents/ folder), the
    placement rule declares the stage file and stitch_sample places the sample's positions, selected from it; no
    documents/maf/ copy is written."""
    import starfinder.stitching as stitching
    from starfinder.io import save_volume
    from .stitching_fixtures import STAGE_SPACING, grid
    stage = tmp_path / "acquisition" / "positions.maf"
    stage.parent.mkdir()
    # Sample s1 (PositionIDs 1, 2) and sample s2 (7, 3), in one acquisition file, in the order 7, 1, 3, 2.
    stage.write_text(maf_text(((7, "0.07", "0.04"), (1, "0.076", "0.048"), (3, "0.07", "0.040261"),
                               (2, "0.076", "0.048261"))))
    positions = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "stage_file": str(stage)}
    config = python_example(tmp_path, stitching=dict(MULTIVIEW, positions=positions), voxel_size_xy=12.0,
                            voxel_size_z=1.0)
    path = write_fovs(config, rows_text())
    output = tmp_path / "example" / "direct"
    names = common_names(config, monkeypatch, checks=True)
    text = (ROOT / "workflow/rules/stitching-py.smk").read_text()
    exec(compile(text[:text.index("if STITCHING_METHOD == 'bigstitcher':")], "stitching-py.smk", "exec"), names)
    assert names["get_stitching_positions"](Wildcards(sample="s1")) == [str(stage)]
    assert names["get_stitching_images"](Wildcards(sample="s1")) == [f"{output}/fovs/F2/reference.tif",
                                                                     f"{output}/fovs/F1/reference.tif"]
    assert names["MEMBERSHIP"] == str(path)
    for fov in ("F2", "F1"):
        save_volume(np.zeros((2, 16, 24), np.uint16), output / "fovs" / fov / "reference.tif",
                    metadata=grid((2, 16, 24), STAGE_SPACING).metadata)
    tables = []
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, **kw: tables.append(table) or "result")
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: None)
    _snakemake._run_stitching(SimpleNamespace(config=config, input=[names["MEMBERSHIP"]],
                                              output=[str(output / "stitching/s1/placement.json")],
                                              wildcards=Wildcards(sample="s1")))
    table, = tables
    assert table.fov_ids == ["F1", "F2"] and table.positions["position_id"].tolist() == [1, 2]   # file order
    assert table.positions["nominal_y"].tolist() == [0.0, 261.0] and table.positions["nominal_x"].tolist() == [0, 0]
    assert not (output / "documents").exists()
    # Without the position_id column the stage file names no FOV of the workflow: the parse raises.
    write_fovs(config, rows_text(positions=False))
    with pytest.raises(ValueError, match="position_id column of fovs.csv"):
        common_names(config, monkeypatch, checks=True)
