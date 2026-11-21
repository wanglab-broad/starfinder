"""The pages of the segmentation input recipe: row MG4 and the validation page's table of implemented checks (W-891).

MG4 of design/validation/segmentation-input-recipe.md: every row of the specification's migration table
(design/specs/segmentation-input-recipe.md, label ``segmentation-input-recipe-migration``) is a row of
docs/migration.md. The validation page lists every check of its tables (label
``segmentation-input-recipe-checks``) in "Implemented checks" (label
``segmentation-input-recipe-implemented-checks``), and every test it names is a test function of the file it names.
A docs-contract module: it reads pages only and holds no slow, learned or extended test.
"""
import ast
import re
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.segmentation, pytest.mark.contract]

TEST_DIR = Path(__file__).resolve().parent
CHECK = re.compile(r"^\| ([A-Z]{2}\d+[ab]?) \|", re.MULTILINE)


def test_mg4_every_migration_row_is_in_the_guide():
    spec = docs_pages.table(docs_pages.design("specs/segmentation-input-recipe.md"),
                            "segmentation-input-recipe-migration").splitlines()
    guide = docs_pages.table(docs_pages.read("migration.md"), "migration-input-recipe-rows").splitlines()
    assert spec[0] == "| Before | After |" and len(spec) == 13
    missing = [row for row in spec[2:] if row not in guide]
    assert not missing, missing


def _tests(path):
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_")}


def test_every_check_has_a_row_naming_its_tests():
    page = docs_pages.design("validation/segmentation-input-recipe.md")
    checks = CHECK.findall(docs_pages.section(page, "segmentation-input-recipe-checks"))
    rows = docs_pages.table(page, "segmentation-input-recipe-implemented-checks").splitlines()[2:]
    listed = {}
    for row in rows:
        check, files, tests = (cell.strip() for cell in row.strip().strip("|").split("|"))
        listed[check] = (files, tests)
    assert sorted(listed) == sorted(checks) and len(checks) == 50
    for check, (files, tests) in listed.items():
        if check == "PF3":                                           # the controller's a19-parity check
            assert "a19-parity" in files
            continue
        for file, names in zip(files.split(";"), tests.split(";")):
            defined = _tests(TEST_DIR / file.strip().strip("`"))
            for name in (n.strip().strip("`") for n in names.split(",")):
                assert name in defined, (check, file, name)
