"""The table "Implemented checks" of docs/stitching-assembly-validation.md and the rows it maps (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): ``test_every_row_names_collected_tests``, from
``test_stitching_validation.py``, reads the page here, so that no ``slow`` test reads it. It finds the rows of the
page's check tables and of the real examples in the passage from the label ``(stitching-assembly-validation-checks)=``
to the label ``(stitching-assembly-validation-implemented-checks)=``, and the table by the second label
(``docs_pages``), not by headings: the table has exactly those rows (each case line of LB7 a row of its own) and the
RX rows; every row but an RX row names tests that pytest collects; none is marked as an expected failure, none is
skipped where the extras are importable, and the tier column agrees with the ``slow`` marks.
"""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

from . import docs_pages
from .conftest import SUBSYSTEM_MARKERS

pytestmark = [pytest.mark.stitching, pytest.mark.contract]

TEST_ROOT = Path(__file__).resolve().parent
PAGE = "stitching-assembly-validation.md"
ROW = r"(?:PL|MV|AS|SC|EE|EX|FI|LB|SD|WF|RX)\d+(?:\.\d+)?"
BATCH_EXTRAS = ("multiview_stitcher", "spatialdata", "anndata")


# --- The table of the section "Implemented checks" ------------------------------------------------

def design_rows():
    """The row ids of the page's check tables and of the real examples: the passage from the label of "Checks" to the
    label of the table "Implemented checks"."""
    text = docs_pages.between(docs_pages.read(PAGE), "stitching-assembly-validation-checks",
                              "stitching-assembly-validation-implemented-checks")
    return [m.group(1) for m in re.finditer(rf"^\| ({ROW}) \|", text, flags=re.MULTILINE)]


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, the cell, its tier)}."""
    text = docs_pages.table(docs_pages.read(PAGE), "stitching-assembly-validation-implemented-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(rf"\| ({ROW}) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(2).strip(),
                                    match.group(3).strip())
    return rows


def named_tests(names):
    """(module, test name) of a row, each name under the module named before it."""
    module, named = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            named.append((module, name))
    return named


def collected(session, module):
    """The items pytest collects from one test module (its classes included)."""
    items, nodes = [], [pytest.Module.from_parent(session, path=TEST_ROOT / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def matches(item, name):
    if name.endswith("*"):
        return item.originalname.startswith(name[:-1])
    if "[" in name:
        return item.name == name
    return item.originalname == name


def runtime_skips(module):
    """The modules each ``importorskip`` call of a test module names, and whether it calls ``pytest.skip``."""
    imported, skips = set(), False
    for node in ast.walk(ast.parse((TEST_ROOT / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "importorskip" and node.args and isinstance(node.args[0], ast.Constant):
                imported.add(node.args[0].value)
            skips |= node.func.attr == "skip"
    return imported, skips


def skip_reason(item):
    """The reason the marks of a collected item skip it here, or None."""
    for mark in item.iter_markers("skip"):
        return mark.kwargs.get("reason", "unconditional skip")
    for mark in item.iter_markers("skipif"):
        conditions = mark.args or (mark.kwargs["condition"],)
        assert not any(isinstance(c, str) for c in conditions), item.nodeid
        if any(conditions):
            return mark.kwargs.get("reason", "skipif")
    return None


def test_every_row_names_collected_tests(request):
    rows = implemented_checks()
    expected = design_rows()
    assert len(expected) == len(set(expected))
    assert {f"LB7.{k}" for k in range(1, 12)} <= set(expected)
    assert list(rows) == expected
    batch = all(importlib.util.find_spec(name) is not None for name in BATCH_EXTRAS)
    modules = sorted({module for names, _, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for module in modules:
        imported, skips = runtime_skips(module)
        assert not skips, module
        missing = sorted(m for m in imported if importlib.util.find_spec(m) is None)
        assert not (batch and missing), (module, missing)
    for row, (names, cell, tier) in rows.items():
        named = named_tests(names)
        if row.startswith("RX"):
            assert named == [] and "real-example issue" in cell and tier == "real examples", row
            continue
        assert named, row
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if matches(item, name)]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert "xfail" not in marks, item.nodeid
            assert SUBSYSTEM_MARKERS & marks, item.nodeid
            reason = skip_reason(item)
            if batch:
                assert reason is None, (row, item.nodeid, reason)
            elif reason is not None:
                assert "extra" in reason, (row, item.nodeid, reason)
        slow = [item.get_closest_marker("slow") is not None for item in items]
        if tier == "default":
            assert not any(slow), row
        elif tier == "`slow`":
            assert all(slow), row
