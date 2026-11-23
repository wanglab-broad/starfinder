"""The validation page of the deconvolution stage maps every check to its tests (W-923).

design/validation/deconvolution.md lists the checks in its tables (label ``deconvolution-checks``) and, in "Implemented
checks" (label ``deconvolution-implemented-checks``), the files and the tests of each; every test it names is a test
function of the file it names, under ``src/python/test/``. A docs-contract module: it reads pages only.
"""
import ast
import re
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

TESTS = Path(__file__).resolve().parent
CHECK = re.compile(r"^\| ([A-Z]{2}\d+b?) \|", re.MULTILINE)
TIME = re.compile(r"^(—|\d+\.\d|skips here \([DP]\))$")


def _tests(path):
    """The test functions of a file."""
    return {node.name for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def test_every_check_has_a_row_naming_its_tests():
    page = docs_pages.design("validation/deconvolution.md")
    checks = CHECK.findall(docs_pages.section(page, "deconvolution-checks").split("## Image checks")[0])
    assert len(checks) == 56 and len(set(checks)) == 56
    rows = docs_pages.table(page, "deconvolution-implemented-checks").splitlines()
    assert rows[0] == "| Check | File | Tests | Time (s, one CPU) |"
    listed = {}
    for row in rows[2:]:
        check, files, tests, time = (cell.strip() for cell in row.strip().strip("|").split("|"))
        assert check not in listed, check
        assert TIME.match(time), (check, time)
        listed[check] = (files, tests)
    assert sorted(listed) == sorted(checks)
    for check, (files, tests) in listed.items():
        for file, names in zip(files.split(";"), tests.split(";"), strict=True):
            defined = _tests(TESTS / file.strip().strip("`"))
            for name in (n.strip().strip("`") for n in names.split(",")):
                assert name in defined, (check, file, name)
