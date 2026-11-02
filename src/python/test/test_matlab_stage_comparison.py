"""Python stage outputs against the committed MATLAB reference outputs on a literal fixture (W-212).

design/validation/matlab-stage-comparison.md fixes the fixture, the mappings and checks D1-N1;
scripts/matlab_stage_comparison.py implements them. The references under test/data/matlab_reference/ come from one
MATLAB R2023b run of workflow/scripts/compare_matlab_stages.m through the path launcher; MATLAB is not needed here.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.integration, pytest.mark.golden]

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("matlab_stage_comparison",
                                              ROOT / "src/python/scripts/matlab_stage_comparison.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


@pytest.fixture(scope="module")
def reference():
    return tool.load_reference()


@pytest.fixture(scope="module")
def python():
    return {depth: tool.python_stages(depth) for depth in tool.DEPTHS}


def test_references_record_one_real_matlab_run_through_the_path_launcher(reference):
    provenance = reference["provenance"]
    matlab, launcher = provenance["matlab"], provenance["launcher"]
    assert matlab["release"] == "2023b" and "R2023b" in matlab["version"] and matlab["threads"] == 1
    assert matlab["licenses"] == {"Image_Toolbox": 1, "Statistics_Toolbox": 1}
    assert set(tool.TOOLBOXES) <= {t["name"] for t in matlab["toolboxes"]}
    assert (launcher["matlab_launcher"], launcher["matlab_single_thread"], launcher["entry"]) == \
        ("path", True, "compare_matlab_stages")
    command = launcher["command"]
    assert command[0].endswith("/bin/matlab") and command[1:3] == ["-singleCompThread", "-batch"]
    assert command[3].startswith("addpath(") and "compare_matlab_stages('" in command[3]
    assert len(provenance["protocol"]["sha256"]) == 64 and provenance["protocol"]["path"] == str(tool.PROTOCOL)
    # The references belong to today's fixture: changing it fails here until they are regenerated.
    assert provenance["fixture_sha256"] == tool.fixture_sha256()
    assert all(reference[f"z{d}"]["fixture_sha256"] == tool.fixture_sha256() for d in tool.DEPTHS)
    assert set(provenance["raw_sha256"]) == set(tool.RAW_OUTPUTS)


@pytest.mark.parametrize("depth", tool.DEPTHS)
def test_python_stages_match_the_matlab_reference(python, reference, depth):
    row = tool.compare_depth(python[depth], reference[f"z{depth}"])
    assert row["registration"]["correction_zyx"] == [0, -1, 1]
    assert [r["calls"][:2] for r in row["readout"]] == [["12", "21"]] * 2
    assert [r["genes"] for r in row["readout"]] == [["gene-A", "gene-B"]] * 2
    if depth == 1:
        assert row["detection"].startswith("not compared")
    else:
        assert row["detection"] == "compared"


@pytest.mark.parametrize("broken, stage", [
    (dict(point_axes=(0, 1, 2)), "detection"),                 # MATLAB XYZ read as ZYX
    (dict(index_base=0), "detection"),                         # one-based indices kept: a one-voxel offset
    (dict(image_axes=(0, 1, 2)), "registration"),              # MATLAB YXZ image read as ZYX
    (dict(shift_order=(0, 1, 2)), "registration"),             # MATLAB [row, column, z] read as ZYX
    (dict(extraction_channels=(1, 2, 3, 4)), "extraction"),    # the channel reorder [2 1 4 3] not applied
])
def test_a_mapping_error_fails_the_comparison(python, reference, broken, stage):
    with pytest.raises(AssertionError, match=stage):
        tool.compare_depth(python[5], reference["z5"], replace(tool.MAPPING, **broken))


@pytest.mark.contract
def test_the_protocol_record_names_the_tool_mappings():
    text = docs_pages.design("validation/matlab-stage-comparison.md")
    mapping = docs_pages.table(text, "matlab-stage-mapping")
    assert "transposed `(1, 2, 0, 3)`" in mapping and "transpose `(2, 0, 1)`" in mapping
    assert "`[2 1 4 3]`" in mapping and "`{1: 1, 2: 0, 3: 3, 4: 2}`" in mapping
    assert tool.MAPPING.extraction_channels == (2, 1, 4, 3) and tool.COLOR_TO_CHANNEL == {"1": 1, "2": 0, "3": 3, "4": 2}
    checks = docs_pages.table(text, "matlab-stage-checks")
    assert [line.split("|")[1].strip() for line in checks.splitlines()[2:]] == \
        ["D1", "D2", "R1", "R2", "E1", "E2", "E3", "F1", "F2", "N1"]
    assert "`1e-9`" in checks and checks.count("`atol=1e-12, rtol=0`") == 2
    assert (tool.DISTANCE, tool.ATOL) == (1e-9, 1e-12)
