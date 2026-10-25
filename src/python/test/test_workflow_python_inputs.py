"""What the Python workflow rules take and require (W-569: OM-02, OM-15, OM-11).

The Python rules take no JSON copy of the configuration and name their inputs, which the adapters read by name; the
MATLAB rules keep the copy, which their scripts read. A Python configuration needs fiji_path, envs_path, img_row and
img_col only where a rule reads them: Fiji in the legacy BigStitcher rule (stitching.smk, included for the MATLAB
backend only), the dimensions in the subtile windows and the legacy tile configuration writer.
"""

import copy
import re
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from starfinder.dataset import from_workflow_config
from starfinder.dataset.workflow import _stitching_block

from .test_dataset_fov_layout import backend_namespace, rule_section
from .test_fov_membership import common_names, rows_text, write_fovs
from .test_stitching_workflow import ASSEMBLY, STITCHING
from .test_workflow_config_validation import snakefile_errors

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "workflow" / "rules"
SCHEMA = yaml.safe_load((ROOT / "workflow" / "schemas" / "config.schema.yaml").read_text())
# The processing rules of each backend: (rule file, rule) of the Python and the MATLAB backend.
PROCESSING = {"rsf_single_fov": ("registration-py.smk", "registration.smk"),
              "gr_single_fov_subtile": ("registration-py.smk", "registration.smk"),
              "deep_create_subtile": ("registration-py.smk", "spot-finding.smk"),
              "nuclei_registration": ("registration-py.smk", "registration.smk"),
              "lrsf_single_fov_subtile": ("spot-finding-py.smk", "spot-finding.smk"),
              "deep_rsf_subtile": ("spot-finding-py.smk", "spot-finding.smk")}
# The named inputs of each Python rule, which _run_workflow reads by name (codebook, subtile).
NAMED = {"rsf_single_fov": {"codebook", "images"}, "gr_single_fov_subtile": {"codebook", "images"},
         "deep_create_subtile": {"images"}, "nuclei_registration": {"images"},
         "lrsf_single_fov_subtile": {"codebook", "subtile"}, "deep_rsf_subtile": {"codebook", "subtile"}}


def namespace(tmp_path, backend):
    names, _ = backend_namespace(tmp_path, backend, config_path=str(tmp_path / "config.yaml"))
    from starfinder._snakemake import _image_dir_inputs
    return dict(names, CODEBOOK=str(tmp_path / "in" / "genes.csv"), _image_dir_inputs=_image_dir_inputs,
                additional_round_names=["flamingo"])


def test_the_python_rules_take_no_json_copy_and_the_matlab_rules_keep_it(tmp_path):
    """Each processing rule parsed for its backend: the MATLAB rule's first input is the JSON copy, which its script
    reads; the Python rule's inputs are named and none of them is the copy."""
    json_copy = str(tmp_path / "config.json")
    python, matlab = namespace(tmp_path, "python"), namespace(tmp_path, "matlab")
    for rule, (python_file, matlab_file) in PROCESSING.items():
        args, kwargs = rule_section(RULES / python_file, rule, "input", python)
        assert args == () and set(kwargs) == NAMED[rule], rule
        # The inputs the adapter reads are one path each, so that snakemake.input.<name> is a string.
        assert all(isinstance(kwargs[name], str) for name in ("codebook", "subtile") if name in kwargs), rule
        files = [f for value in kwargs.values() for f in (value if isinstance(value, list) else [value])]
        assert json_copy not in files and not [f for f in files if f.endswith(".json")], rule
        matlab_args, _ = rule_section(RULES / matlab_file, rule, "input", matlab)
        assert matlab_args[0] == json_copy, rule
    # The copy's rule exists for the MATLAB backend only; the Python block of common.smk never names it.
    common = (RULES / "common.smk").read_text()
    assert re.search(r'^if BACKEND == "matlab":\n    rule rsf_preparation:\n', common, re.M)
    assert not re.search(r"^rule rsf_preparation:", common, re.M)
    assert 'if BACKEND == "matlab":\n    localrules: create_sample_maf, rsf_preparation' in \
        (ROOT / "workflow" / "Snakefile").read_text()
    for name in ("registration-py.smk", "spot-finding-py.smk"):
        assert ".replace('.yaml', '.json')" not in (RULES / name).read_text(), name
    adapters = (ROOT / "src" / "python" / "starfinder" / "_snakemake.py").read_text()
    run_workflow = adapters[adapters.index("def _run_workflow("):adapters.index("def _run_nuclei_registration(")]
    assert "snakemake.input[" not in run_workflow
    assert "snakemake.input.codebook" in run_workflow and "snakemake.input.subtile" in run_workflow


def minimal(tmp_path, **change):
    """The minimal Python example without envs_path, fiji_path, img_row and img_col, under tmp_path."""
    config = yaml.safe_load((ROOT / "docs" / "examples" / "workflow-minimal.yaml").read_text())
    for key in ("envs_path", "fiji_path", "img_row", "img_col"):
        config.pop(key, None)
    config.update(root_input_path=str(tmp_path / "input"), root_output_path=str(tmp_path / "output"),
                  starfinder_path=str(ROOT), **change)
    return config


def test_a_python_configuration_without_fiji_envs_and_dimensions_parses(tmp_path, monkeypatch):
    """Without a legacy BigStitcher rule and without subtile windows, the configuration passes the Snakefile's
    validation, the adapters and the parse-time code of common.smk; no Python rule file reads the four keys."""
    config = minimal(tmp_path)
    write_fovs(config, rows_text(positions=False))
    assert snakefile_errors(copy.deepcopy(config)) == []
    assert from_workflow_config(config).dataset.subtile is None
    names = common_names(config, monkeypatch, checks=True)
    assert names["BACKEND"] == "python" and names["FOVS"] == ["F10", "F2", "F3", "F1"]
    for name in ("common.smk", "registration-py.smk", "spot-finding-py.smk", "stitching-py.smk", "segmentation.smk",
                 "reads-assignment.smk", "export.smk", "report.smk"):
        text = (RULES / name).read_text()
        assert not re.search(r"config\[['\"](fiji_path|envs_path|img_row|img_col)['\"]\]", text), name
    # Fiji is read when the rule file is parsed only by the legacy BigStitcher rule, which only MATLAB includes.
    assert "config['fiji_path']" in (RULES / "stitching.smk").read_text()
    assert re.search(r'elif BACKEND == "matlab":\n\s+include: "rules/stitching.smk"',
                     (ROOT / "workflow" / "Snakefile").read_text())


REQUIRED = ["'img_col' is a required property", "'img_row' is a required property"]


def test_the_dimensions_are_required_by_their_consumers_only(tmp_path):
    """img_row and img_col: required by an active create_subtiles block and by the legacy tile configuration writer
    (stitching method bigstitcher without an assembly block), in the schema and in the adapters."""
    subtile = minimal(tmp_path, rules={"rsf_single_fov": {"run": True, "parameters": {}},
                                       "gr_single_fov_subtile": {"run": True, "parameters": {
                                           "create_subtiles": {"run": True, "sqrt_pieces": 2}}}})
    assert sorted(e.message for e in snakefile_errors(copy.deepcopy(subtile))) == REQUIRED
    for rule in ("rsf_single_fov", "gr_single_fov_subtile"):
        with pytest.raises(ValueError, match=r"img_row and img_col \(the image height and width in pixels\) are "
                                             r"required by rules\.gr_single_fov_subtile\.parameters\.create_subtiles"):
            from_workflow_config(subtile, rule)
    sized = dict(subtile, img_row=32, img_col=32)
    assert snakefile_errors(copy.deepcopy(sized)) == []
    assert from_workflow_config(sized, "gr_single_fov_subtile").dataset.subtile.windows
    inactive = copy.deepcopy(subtile)
    inactive["rules"]["gr_single_fov_subtile"]["parameters"]["create_subtiles"] = {"run": False}
    assert snakefile_errors(copy.deepcopy(inactive)) == []

    stitched = minimal(tmp_path, stitching=copy.deepcopy(STITCHING))
    assert sorted(e.message for e in snakefile_errors(copy.deepcopy(stitched))) == REQUIRED
    with pytest.raises(ValueError, match="required by the legacy tile configuration writer"):
        _stitching_block(stitched)
    assert _stitching_block(dict(stitched, img_row=32, img_col=32)).method == "bigstitcher"
    assembled = dict(stitched, assembly=copy.deepcopy(ASSEMBLY))
    assert snakefile_errors(copy.deepcopy(assembled)) == []
    assert _stitching_block(assembled).method == "bigstitcher"
    # The MATLAB backend keeps them required (its legacy stitching scripts read them).
    assert "img_col" in SCHEMA["allOf"][0]["else"]["required"] and "img_row" in SCHEMA["allOf"][0]["else"]["required"]


def test_the_native_path_does_not_import_the_legacy_translation(tmp_path):
    """A Python configuration is read without importing starfinder.dataset._legacy_config; a MATLAB-backend one, which
    the adapters of the shared rules read, imports it, and workflow.py imports it only inside that branch."""
    script = """
import sys, yaml
from starfinder.dataset import from_workflow_config
config = yaml.safe_load(open(sys.argv[1]))
from_workflow_config(config)
print('python', 'starfinder.dataset._legacy_config' in sys.modules)
config['rules']['rsf_single_fov']['parameters'] = {'global_registration': {'run': True}}
del config['backend']
from_workflow_config(config)
print('matlab', 'starfinder.dataset._legacy_config' in sys.modules)
"""
    example = ROOT / "docs" / "examples" / "workflow-minimal.yaml"
    result = subprocess.run([sys.executable, "-c", script, str(example)], capture_output=True, text=True, cwd=tmp_path)
    assert result.stdout.split() == ["python", "False", "matlab", "True"], result.stderr
    workflow = (ROOT / "src" / "python" / "starfinder" / "dataset" / "workflow.py").read_text()
    assert not re.search(r"^from \.?_legacy_config|^import .*_legacy_config", workflow, re.M)
    assert re.findall(r"^\s+from \._legacy_config import (\w+)", workflow, re.M) == ["legacy_processing",
                                                                                "matlab_reference_stains"]
