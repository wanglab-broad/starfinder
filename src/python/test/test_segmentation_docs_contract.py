"""The table "Implemented checks" of design/specs/assignment-algorithms.md, which maps the §2.9 rows to their tests (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): ``test_every_l_row_names_existing_tests`` (from
``test_segmentation_validation.py``) and ``test_every_a_row_names_existing_tests`` (from
``test_assignment_validation.py``) read the page here, so that no ``slow`` or ``learned`` test reads it. They find
the table by its label ``(assignment-algorithms-implemented-checks)=`` (``docs_pages``), not by its heading.
"""
import ast
import re
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.segmentation, pytest.mark.contract]

TEST_ROOT = Path(__file__).resolve().parent
PAGE = "specs/assignment-algorithms.md"


def implemented_checks():
    """{row: the backticked names of its "Module and tests" cell} from the design record."""
    text = docs_pages.table(docs_pages.design(PAGE), "assignment-algorithms-implemented-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| ([LA]\d+) \| ([^|]+) \|", line)
        if match:
            rows[match.group(1)] = re.findall(r"`([^`]+)`", match.group(2))
    return rows


def defined_tests(module):
    tree = ast.parse((TEST_ROOT / module).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def unmatched_names(names):
    """The test names of one row that no function of the module named before them matches."""
    module, missing = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
            continue
        if not name.startswith("test_"):
            continue
        pattern = name.split("[", 1)[0]
        defined = defined_tests(module) if module else set()
        found = any(f.startswith(pattern[:-1]) for f in defined) if pattern.endswith("*") else pattern in defined
        if not found:
            missing.append((module, name))
    return missing


def check_rows(prefix, count):
    rows = implemented_checks()
    expected = [f"{prefix}{i}" for i in range(1, count + 1)]
    assert sorted(r for r in rows if r.startswith(prefix)) == sorted(expected)
    for row in expected:
        # A test name, not the module's file name, which starts with test_ too (W-602).
        assert any(n.startswith("test_") and not n.endswith(".py") for n in rows[row]), row
        assert unmatched_names(rows[row]) == [], row


def test_every_l_row_names_existing_tests():
    check_rows("L", 14)


def test_every_a_row_names_existing_tests():
    check_rows("A", 20)
