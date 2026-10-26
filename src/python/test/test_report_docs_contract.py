"""The tables of docs/report-algorithms.md that the §2.11 report's tests parse (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): it holds the tests that read the page, so that no
``slow`` test of the report reads it, and it finds each part by its label (``docs_pages``), not by its heading:

* ``(report-algorithms-checks)=`` and ``(report-algorithms-implemented-checks)=``: the tables "Checks" and
  "Implemented checks (W-459)". ``test_every_check_names_collected_tests`` keeps the second true; it was in
  ``test_report_chain.py``.
* ``(report-algorithms-figure-sizes)=``: the section "Figure sizes", with its table of slots, its rules and its
  table of figures. Its two tests are the page's part of ``test_slots_are_the_page_columns_and_the_table``
  (``test_report_figure_rules.py``) and of ``test_figures_table_equals_the_constants`` (``test_report_second_review.py``),
  which check the code against ``report_sizes``, the numbers written from this section.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from . import docs_pages
from .conftest import SUBSYSTEM_MARKERS
from .report_sizes import COMPARISON, FIGURES, FULL, HALF, THIRD

pytestmark = [pytest.mark.report, pytest.mark.contract]

PAGE = "report-algorithms.md"
ROW = r"(?:RC|DF|PR|EF|RP|PG|CH|RT|Z|A|CP|RR|WF|DT)\d+"
TIERS = {"default": (False,), "slow": (True,), "default and slow": (False, True)}


# --- The table "Implemented checks (W-459)" -------------------------------------------------------------------------

def design_rows():
    """The row ids of the table "Checks" of the validation design."""
    text = docs_pages.table(docs_pages.read(PAGE), "report-algorithms-checks")
    return [m.group(1) for m in re.finditer(rf"^\| ({ROW}) \|", text, flags=re.MULTILINE)]


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, its tier)}."""
    text = docs_pages.table(docs_pages.read(PAGE), "report-algorithms-implemented-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(rf"\| ({ROW}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(3).strip())
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
    """The items pytest collects from one test module."""
    items, nodes = [], [pytest.Module.from_parent(session, path=Path(__file__).resolve().parent / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def test_every_check_names_collected_tests(request):
    """Every row of the table "Checks" has a row in "Implemented checks (W-459)", in the same order, naming tests that
    pytest collects, none an expected failure, each with a subsystem marker and the tier the row states."""
    rows, expected = implemented_checks(), design_rows()
    assert len(expected) == len(set(expected)) == 54
    assert list(rows) == expected
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
        assert named and tier in TIERS, row
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if (item.originalname.startswith(name[:-1])
                                                           if name.endswith("*") else item.originalname == name)]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert "xfail" not in marks and SUBSYSTEM_MARKERS & marks, item.nodeid
        slow = {item.get_closest_marker("slow") is not None for item in items}
        assert tuple(sorted(slow)) == TIERS[tier], (row, tier)


# --- The section "Figure sizes" -------------------------------------------------------------------------------------

def numbers(text):
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]


def test_the_slot_table_of_figure_sizes_is_the_slots():
    """The page's table of slots has one row per slot with its width in px, the widths of ``report_sizes``, and the
    rules state the sizes the figure tests check."""
    passage = docs_pages.section(docs_pages.read(PAGE), "report-algorithms-figure-sizes")
    table = dict(re.findall(r"^\| ([A-Za-z ,'()]+?) \| ([\d,]+) \|", passage, flags=re.M))
    assert {k: int(v.replace(",", "")) for k, v in table.items()} == {
        "Full column": FULL, "Half column": HALF, "Third of the column": THIRD, "Comparison page column": COMPARISON}
    for words in ("twice the width of its slot", "11 px", "8 px", "14 px", "64"):
        assert words in passage, words


def test_the_figure_table_of_figure_sizes_is_report_sizes():
    """The page's table of figures has the ten rows of ``report_sizes.FIGURES``, each with its slot, its shown widths
    and the leading numbers of its drawn width, and the rules say "two thirds"."""
    passage = docs_pages.section(docs_pages.read(PAGE), "report-algorithms-figure-sizes")
    rows = re.findall(r"^\| ([^|]+?) \| (full|half|third|third or half|comparison) \| ([\d, or]+?) \| ([^|]+?) \|",
                      passage, flags=re.M)
    table = {name: (slot, numbers(shown_w), numbers(drawn)) for name, slot, shown_w, drawn in rows}
    assert len(table) == 10 and set(table) == set(FIGURES)
    for name, (slot, shown_w, drawn) in FIGURES.items():
        assert table[name][:2] == (slot, shown_w), name
        assert table[name][2][:len(drawn)] == drawn, name
    assert "two thirds" in passage
