"""Keep the implemented-checks table aligned with the collected synthetic tests.

``test_every_ga_row_names_collected_tests`` reads the generator-check rows and checks that
"Implemented checks" has exactly those rows, in order; that each names a collected test;
that no named test is skipped or expected to fail; that each carries the listed subsystem
and kind markers; and that its tier matches the tests' ``slow`` markers. A row whose tier is
``evidence project`` names its script under ``benchmarks/`` and no test (W-536); every
script a row names exists.
"""
from pathlib import Path
import re

import pytest


pytestmark = [pytest.mark.synthetic]

TEST = Path(__file__).resolve().parent
REPOSITORY = TEST.parents[2]
DESIGN = TEST.parents[2] / "docs" / "synthetic-real-appearance-validation.md"
SECTION = "## Implemented checks\n"
TIERS = {"routine": (False,), "slow": (True,), "routine and slow": (False, True)}
# The tier of a check that the Chapter II evidence project runs from its script under benchmarks/ (W-536).
EVIDENCE = "evidence project"
KINDS = {"contract", "validation", "golden"}

# --- The table "Implemented checks" -----------------------------------------------------------------------------------

def ga_rows():
    """{row: the backticked markers of its "Tier and markers" cell} of the table "Generator checks", in page order."""
    text = DESIGN.read_text(encoding="utf-8").split("## Generator checks\n", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| (GA\d+) \|", line)
        if match:
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            assert len(cells) == 7 and match.group(1) not in rows, line[:40]
            rows[match.group(1)] = set(re.findall(r"`([a-z_]+)`", cells[6]))
    return rows


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, its tier)}, in table order."""
    section = DESIGN.read_text(encoding="utf-8").split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"\| (GA\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(4).strip())
    return rows


def named_tests(names):
    """(module, test name) of a row, each test under the module named before it."""
    module, named = None, []
    for name in names:
        if name.startswith("benchmarks/"):
            continue
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            named.append((module, name))
    return named


def collected(session, module):
    """The items pytest collects from one test module."""
    items, nodes = [], [pytest.Module.from_parent(session, path=TEST / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


@pytest.mark.contract
def test_every_ga_row_names_collected_tests(request):
    expected, rows = ga_rows(), implemented_checks()
    expected_rows = [f"GA{k}" for k in range(1, 25)] + ["GA26", "GA27"]
    assert list(expected) == expected_rows
    assert list(rows) == list(expected)
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
        scripts = [name for name in names if name.startswith("benchmarks/")]
        assert all((REPOSITORY / script).is_file() for script in scripts), (row, scripts)
        if tier == EVIDENCE:
            assert scripts and not named, row
            continue
        assert named and tier in TIERS, row
        kinds = expected[row] & KINDS
        assert len(kinds) == 1, (row, kinds)
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if item.originalname == name]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert not marks & {"xfail", "skip", "skipif"}, item.nodeid
            assert "synthetic" in marks and marks & KINDS == kinds, (row, item.nodeid)
        slow = {item.get_closest_marker("slow") is not None for item in items}
        assert tuple(sorted(slow)) == TIERS[tier], (row, tier)
