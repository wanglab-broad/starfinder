"""The validation page of the compressed image outputs maps every check to its tests (W-922).

design/validation/compressed-image-outputs.md lists the package checks in its tables (label
``compressed-image-outputs-checks``) and, in "Implemented checks" (label
``compressed-image-outputs-implemented-checks``), the file and the tests of each; every test it names is a test
function of the file it names (a test under ``src/python/test/`` or an evidence script under ``benchmarks/``). A
docs-contract module: it reads pages only.
"""
import ast
import re
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
CHECK = re.compile(r"^\| ([A-Z]{2}\d+) \|", re.MULTILINE)


def _tests(path):
    """The test functions of a file, and its test methods as ``Class::method``."""
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            found |= {f"{node.name}::{item.name}" for item in node.body
                      if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")}
    return found


def test_every_check_has_a_row_naming_its_tests():
    page = docs_pages.design("validation/compressed-image-outputs.md")
    checks = CHECK.findall(docs_pages.section(page, "compressed-image-outputs-checks"))
    assert len(checks) == 21 and len(set(checks)) == 21
    rows = docs_pages.table(page, "compressed-image-outputs-implemented-checks").splitlines()
    assert rows[0] == "| Check | File | Tests | Time (s, one CPU) |"
    listed = {}
    for row in rows[2:]:
        check, files, tests, _ = (cell.strip() for cell in row.strip().strip("|").split("|"))
        assert check not in listed, check
        listed[check] = (files, tests)
    assert sorted(listed) == sorted(checks)
    for check, (files, tests) in listed.items():
        if check == "ID5":                       # the specification's table, run twice (the run folder's step)
            assert files == "the specification's table `compressed-image-outputs-tests`"
            docs_pages.table(docs_pages.design("specs/compressed-image-outputs.md"), "compressed-image-outputs-tests")
            continue
        for file, names in zip(files.split(";"), tests.split(";"), strict=True):
            file = file.strip().strip("`")
            path = ROOT / file if file.startswith("benchmarks/") else ROOT / "src" / "python" / "test" / file
            defined = _tests(path)
            for name in (n.strip().strip("`").split("[")[0] for n in names.split(",")):
                assert name in defined, (check, file, name)
