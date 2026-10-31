"""The Python workflow's FOV membership (W-566): ``<input_root>/fovs.csv`` and ``selected_fovs``.

``starfinder.dataset.workflow._membership`` reads ``fovs.csv`` (``sample,fov_id`` and an optional
``position_id``) in row order and restricts it by ``selected_fovs``; the plan reads it (the tests of the plan's
membership, its tasks and the stitch and report commands that read the restricted membership moved to
test_workflow_plan.py, test_workflow_plan_refusals.py, test_workflow_stitch_tasks.py and
test_workflow_snakemake_dryrun.py, W-750). ``common.smk`` keeps the sample annotation, the FOV pattern and the subset
keys of the MATLAB backend, and the legacy reads_assignment adapter reads a FOV's sample from either
(``_fov_sample`` of ``starfinder._legacy_workflow``). Fixtures, all written in the test: the repository's example
configurations with a ``fovs.csv`` (or, for the MATLAB backend, a sample annotation) under ``tmp_path``. The rule file
is executed in part (the names ``common.smk`` defines before its first rule); Snakemake itself never runs.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder._legacy_workflow import _fov_sample

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


# The rsf_single_fov rule of the minimal example in the MATLAB keys (those of the example on the base revision), which a
# MATLAB-backend configuration holds: the example's native keys are Python-only.
MATLAB_RSF = {"run": True, "parameters": {
    "global_registration": {"run": True},
    "spot_finding": {"run": True, "intensity_estimation": "noise", "intensity_threshold": 5.0},
    "reads_extraction": {"run": True, "voxel_size": [1, 2, 2]},
    "reads_filtration": {"run": True, "end_base": "CC", "start_base": "C"}}}


MATLAB_FOV_KEYS = (("fov_id_pattern", "tile_{i}"), ("n_fovs", 1), ("subset_list", []), ("subset_range", False),
                   ("subset_start", 1), ("subset_end", 1), ("subset_random", False), ("n_random_tests", 1))


def test_the_schema_refuses_the_matlab_fov_keys_for_python_and_requires_them_for_matlab(tmp_path):
    """The MATLAB schema requires the FOV keys of backend matlab and refuses the Python-only selected_fovs there;
    the package (the plan's shape check, its Python schema and the membership it reads, W-750) refuses each MATLAB
    FOV key in a Python configuration, accepts selected_fovs and refuses an empty or repeated selection."""
    import jsonschema
    from starfinder.workflow._plan import PlanRefused
    from . import workflow_examples as X
    from . import workflow_tasks as T
    schema = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
    validator = jsonschema.validators.validator_for(schema)(schema)
    config = yaml.safe_load((ROOT / "docs/examples/workflow-minimal.yaml").read_text())
    matlab = dict(config, backend="matlab", fov_id_pattern="tile_{i}", n_fovs=1, subset_list=[], subset_range=False,
                  subset_start=1, subset_end=1, subset_random=False, n_random_tests=1,
                  rules={"rsf_single_fov": MATLAB_RSF})
    validator.validate(matlab)
    for invalid in ({k: v for k, v in matlab.items() if k != "n_fovs"}, dict(matlab, selected_fovs=["tile_1"])):
        assert not validator.is_valid(invalid)
    python = T.dataset_keys(tmp_path, processing=X.PROCESSING)
    assert T.validation_error(dict(python, selected_fovs=["tile_1"])) is None
    for key, value in MATLAB_FOV_KEYS:
        assert f"{key} left the Python configuration" in T.validation_error(dict(python, **{key: value})), key
    T.write_inputs(python, "sample,fov_id\ns1,tile_1\n")
    for selected, match in (([], "nonempty list"), (["tile_1", "tile_1"], r"repeated \['tile_1'\]")):
        with pytest.raises(PlanRefused, match=match):
            T.plan(tmp_path, "selected", dict(python, selected_fovs=selected))


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


def test_a_per_sample_rule_reads_the_selected_fovs_only(tmp_path):
    """The legacy reads_assignment adapter takes a FOV's sample and tile-configuration id from the membership:
    fovs.csv restricted by selected_fovs for the Python backend (its PositionID, or without that column its 1-based
    place among its sample's FOVs of the run). The stitch and report commands that read the sample's selected FOVs:
    test_workflow_stitch_tasks.py, test_the_per_sample_commands_read_the_selected_fovs_only (W-750)."""
    config = config_for(tmp_path, selected_fovs=["F1", "F10"])
    path = write_fovs(config, rows_text())
    assert _fov_sample(path, config, "F10") == ("s2", 7)
    assert _fov_sample(path, config, "F1") == ("s1", 1)
    with pytest.raises(ValueError, match="F2 is in no sample"):
        _fov_sample(path, config, "F2")
    write_fovs(config, rows_text(positions=False))
    # Without position_id the id is the FOV's 1-based place among its sample's FOVs of the run.
    assert _fov_sample(path, dict(config, selected_fovs=None), "F3") == ("s2", 2)
    assert _fov_sample(path, config, "F10") == ("s2", 1)


# --- The legacy sample rules of the MATLAB backend -------------------------------------------------------------

def test_the_snakefile_includes_the_legacy_sample_rules_for_matlab_only():
    # The dispatcher includes rules/legacy.smk, which holds the MATLAB includes and local rules, for backend matlab only.
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    assert 'if config.get("backend") == "matlab":\n\n    include: "rules/legacy.smk"\n' in snakefile
    legacy = (ROOT / "workflow/rules/legacy.smk").read_text()
    assert 'include: "stitching.smk"' in legacy and 'include: "utils.smk"' in legacy
    assert "\nlocalrules: create_sample_maf, rsf_preparation\n" in legacy
    reads = (ROOT / "workflow/rules/reads-assignment.smk").read_text()
    legacy = reads[reads.index('if BACKEND == "matlab":'):]
    assert "    rule create_sample_h5ad:" in legacy and "    rule create_sample_reads_assignment:" in legacy
    assert "rule reads_assignment:" not in legacy
    assert "rule create_sample_maf:" in (ROOT / "workflow/rules/utils.smk").read_text()
