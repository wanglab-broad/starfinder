"""What the workflow rules take and require (W-569: OM-02, OM-15, OM-11), now the MATLAB rule set's (W-750).

The MATLAB rules keep the JSON copy of the configuration, which their scripts read, and the MATLAB schema requires
fiji_path's reader (the legacy BigStitcher rule, stitching.smk, included for the MATLAB backend only) and img_row and
img_col. The Python workflow's tasks name their inputs and read no copy, and need none of these keys (they left the
Python configuration): test_workflow_plan.py, test_the_tasks_read_their_named_inputs_and_no_json_copy and
test_the_plan_needs_no_fiji_envs_or_dimensions, and test_workflow_boundary.py (C8), W-750.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from starfinder.dataset.workflow import _stitching_block

from .test_dataset_fov_layout import backend_namespace, rule_section
from .test_stitching_workflow import ASSEMBLY, STITCHING

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "workflow" / "rules"
SCHEMA = yaml.safe_load((ROOT / "workflow" / "schemas" / "config.schema.yaml").read_text())
# The processing rules of the MATLAB backend and their rule files.
PROCESSING = {"rsf_single_fov": "registration.smk", "gr_single_fov_subtile": "registration.smk",
              "deep_create_subtile": "spot-finding.smk", "nuclei_registration": "registration.smk",
              "lrsf_single_fov_subtile": "spot-finding.smk", "deep_rsf_subtile": "spot-finding.smk"}


def test_the_python_rules_take_no_json_copy_and_the_matlab_rules_keep_it(tmp_path):
    """Each MATLAB processing rule's first input is the JSON copy, which its script reads; the copy's rule
    (rsf_preparation, a local rule) is the MATLAB rule set's, which the dispatcher includes for backend matlab only.
    (The Python workflow's tasks: test_workflow_plan.py, test_the_tasks_read_their_named_inputs_and_no_json_copy.)"""
    json_copy = str(tmp_path / "config.json")
    matlab, _ = backend_namespace(tmp_path, "matlab", config_path=str(tmp_path / "config.yaml"))
    matlab.update(CODEBOOK=str(tmp_path / "in" / "genes.csv"), additional_round_names=["flamingo"])
    for rule, matlab_file in PROCESSING.items():
        matlab_args, _ = rule_section(RULES / matlab_file, rule, "input", matlab)
        assert matlab_args[0] == json_copy, rule
    common = (RULES / "common.smk").read_text()
    assert re.search(r"^rule rsf_preparation:\n", common, re.M)
    assert "\nlocalrules: create_sample_maf, rsf_preparation\n" in (RULES / "legacy.smk").read_text()
    assert 'if config.get("backend") == "matlab":\n\n    include: "rules/legacy.smk"\n' in \
        (ROOT / "workflow" / "Snakefile").read_text()


def test_a_python_configuration_without_fiji_envs_and_dimensions_parses():
    """Fiji is read when the rule file is parsed only by the legacy BigStitcher rule, which only the MATLAB rule set
    includes; no other rule file of it reads fiji_path, envs_path, img_row or img_col at parse time. (A Python
    configuration has none of them: test_workflow_plan.py, test_the_plan_needs_no_fiji_envs_or_dimensions.)"""
    assert "config['fiji_path']" in (RULES / "stitching.smk").read_text()
    assert 'include: "stitching.smk"' in (RULES / "legacy.smk").read_text()
    for name in ("common.smk", "registration.smk", "spot-finding.smk", "segmentation.smk", "reads-assignment.smk"):
        text = (RULES / name).read_text()
        assert not re.search(r"config\[['\"](fiji_path|envs_path|img_row|img_col)['\"]\]", text), name


def test_the_dimensions_are_required_by_their_consumers_only(tmp_path):
    """The MATLAB backend keeps img_row and img_col required (its legacy stitching scripts read them). The translator
    of the stitching block requires them for the legacy tile configuration writer (stitching method bigstitcher
    without an assembly block). (The Python subtile windows that required them are retired, W-735; the legacy tile
    configuration writer of the Python backend too, D6.)"""
    stitched = {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "out",
                "root_input_path": str(tmp_path), "root_output_path": str(tmp_path), "n_rounds": 4,
                "ref_round": "round1", "rules": {}, "stitching": dict(STITCHING)}
    with pytest.raises(ValueError, match="required by the legacy tile configuration writer"):
        _stitching_block(stitched)
    assert _stitching_block(dict(stitched, img_row=32, img_col=32)).method == "bigstitcher"
    assert _stitching_block(dict(stitched, assembly=dict(ASSEMBLY))).method == "bigstitcher"
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
    example = ROOT / "src/python/test/data/workflow_examples/workflow-minimal.yaml"
    result = subprocess.run([sys.executable, "-c", script, str(example)], capture_output=True, text=True, cwd=tmp_path)
    assert result.stdout.split() == ["python", "False", "matlab", "True"], result.stderr
    workflow = (ROOT / "src" / "python" / "starfinder" / "dataset" / "workflow.py").read_text()
    assert not re.search(r"^from \.?_legacy_config|^import .*_legacy_config", workflow, re.M)
    assert re.findall(r"^\s+from \._legacy_config import (\w+)", workflow, re.M) == ["legacy_processing",
                                                                                "matlab_reference_stains"]
