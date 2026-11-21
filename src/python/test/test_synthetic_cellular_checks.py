"""Keep the implemented-checks table aligned with the collected synthetic tests.

``test_every_gn_row_names_collected_tests`` reads the generator-check rows and checks that
"Implemented checks" has exactly those rows, in order; that each names a collected test;
that no named test is skipped or expected to fail; that each carries the listed subsystem
and kind markers; and that its tier matches the tests' ``slow`` markers. A row whose tier is
``evidence project`` names its script under ``benchmarks/`` and no test (W-536); every
script a row names exists. A row whose cases are split names, after the script, the tests the
script defines for them (W-899), which the check reads from the script without importing it.

``test_every_chain_row_names_its_routine_tests_and_its_evidence_check`` reads the table "Implemented chain checks"
(W-603): its rows are the chain checks ``CD0`` to ``CU5``, in order; each names collected tests of
``test_integration_cellular_chain.py``, every parameter of which is the routine tier, and the check of the same name
in ``benchmarks/cellular_chain_evidence.py``, whose cases are those tests; and every chain test is in the table.

``test_the_slow_preset_evidence_script_collects_exactly_its_moved_tests`` (W-899) collects
``benchmarks/synthetic_scene_evidence.py`` (``--list``, in a process of its own): exactly GN7 and GN5 on the three
``slow`` presets and GA1's digests of them, each with the node ID it had in the suite.

A docs-contract module (docs/contributing.md, "Test markers"; W-602): it finds the two tables by their
labels (``docs_pages``), not by their headings: ``(synthetic-cellular-scene-validation-generator-checks)=`` and
``(synthetic-cellular-scene-validation-implemented-checks)=`` and
``(synthetic-cellular-scene-validation-implemented-chain-checks)=``.
"""
import ast
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from . import docs_pages

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

TEST = Path(__file__).resolve().parent
REPOSITORY = TEST.parents[2]
PAGE = "validation/synthetic-cellular-scene.md"
TIERS = {"routine": (False,), "slow": (True,), "routine and slow": (False, True)}
# The tier of a check that the Chapter II evidence project runs from its script under benchmarks/ (W-536).
EVIDENCE = "evidence project"
KINDS = {"contract", "validation", "golden"}

# --- The table "Implemented checks" -----------------------------------------------------------------------------------

def gn_rows():
    """{row: the backticked markers of its "Tier and markers" cell} of the table "Generator checks", in page order."""
    text = docs_pages.table(docs_pages.design(PAGE), "synthetic-cellular-scene-validation-generator-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| (GN\d+) \|", line)
        if match:
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            assert len(cells) == 7 and match.group(1) not in rows, line[:40]
            rows[match.group(1)] = set(re.findall(r"`([a-z_]+)`", cells[6]))
    return rows


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, its tier)}, in table order."""
    text = docs_pages.table(docs_pages.design(PAGE), "synthetic-cellular-scene-validation-implemented-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| (GN\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(4).strip())
    return rows


def named_tests(names, scripts=False):
    """(module, test name) of a row, each test under the module named before it; a test named after a script under
    benchmarks/ is that script's (W-899), and is listed instead of the suite's when scripts is true."""
    module, named = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            if module.startswith("benchmarks/") == scripts:
                named.append((module, name))
    return named


@functools.cache
def script_tests(script):
    """The test functions an evidence script under benchmarks/ defines at its top level (read, not imported)."""
    tree = ast.parse((REPOSITORY / script).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def collected(session, module):
    """The items pytest collects from one test module."""
    items, nodes = [], [pytest.Module.from_parent(session, path=TEST / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def test_every_gn_row_names_collected_tests(request):
    expected, rows = gn_rows(), implemented_checks()
    assert list(expected) == [f"GN{k}" for k in range(1, 25)]
    assert list(rows) == list(expected)
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
        scripts = [name for name in names if name.startswith("benchmarks/")]
        assert all((REPOSITORY / script).is_file() for script in scripts), (row, scripts)
        # A row's part in an evidence script names tests that the script defines (W-899).
        for script, name in named_tests(names, scripts=True):
            assert name in script_tests(script), (row, script, name)
        if tier == EVIDENCE:
            assert scripts and not named, row
            continue
        assert named and tier in TIERS, row
        kinds = expected[row] & KINDS
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if item.originalname == name]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert not marks & {"xfail", "skip", "skipif"}, item.nodeid
            assert "synthetic" in marks and len(marks & KINDS) == 1, item.nodeid
            # The kind marker is one the row names when a kind is specified.
            assert not kinds or marks & kinds, (row, item.nodeid)
        slow = {item.get_closest_marker("slow") is not None for item in items}
        assert tuple(sorted(slow)) == TIERS[tier], (row, tier)


# --- The table "Implemented chain checks" (W-603) --------------------------------------------------------------------

CHAIN_MODULE = "test_integration_cellular_chain.py"
CHAIN_SCRIPT = "benchmarks/cellular_chain_evidence.py"
CHAIN_MARKERS = {"integration", "validation", "slow"}


def chain_rows():
    """{row: (the backticked names of its "Module and tests" cell, those of its evidence cell, the tier the suite
    keeps)}, in table order."""
    text = docs_pages.table(docs_pages.design(PAGE), "synthetic-cellular-scene-validation-implemented-chain-checks")
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| (C[DTU]\d) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), re.findall(r"`([^`]+)`", match.group(3)),
                                    match.group(4).strip())
    return rows


def evidence_script():
    """benchmarks/cellular_chain_evidence.py, loaded by its path (its top level imports the standard library only)."""
    spec = importlib.util.spec_from_file_location("cellular_chain_evidence", REPOSITORY / CHAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_chain_row_names_its_routine_tests_and_its_evidence_check(request):
    script, rows = evidence_script(), chain_rows()
    assert list(rows) == list(script.CHECKS) == [f"C{c}{k}" for c in "DTU" for k in range(6)]
    items = collected(request.session, CHAIN_MODULE)
    of_test = {}
    for item in items:
        of_test.setdefault(item.originalname, []).append(item)
    chain_tests = {name for name, found in of_test.items() if all(hasattr(item, "callspec") for item in found)}
    # The suite keeps the routine tier only; the slow tier is the script's.
    assert {item.callspec.params["tier"] for name in chain_tests for item in of_test[name]} == {"routine"}
    named_all = set()
    for row, (names, evidence, tier) in rows.items():
        named = named_tests(names)
        assert named and {module for module, _ in named} == {CHAIN_MODULE} and tier == "routine", row
        assert evidence == [CHAIN_SCRIPT, row], (row, evidence)
        tests = {name for _, name in named}
        assert tests == {test for check, test, _ in script.CASES if check == row}, row
        for name in tests:
            assert name in chain_tests, (row, name)
            for item in of_test[name]:
                marks = {mark.name for mark in item.iter_markers()}
                assert CHAIN_MARKERS <= marks and not marks & {"xfail", "skip", "skipif"}, item.nodeid
        named_all |= tests
    assert named_all == chain_tests
    # One case per former slow node ID: 23 tests at [slow], four culture tests at [slow-42] and [slow-43].
    assert len({script.node_id(test, seed) for _, test, seed in script.CASES}) == len(script.CASES) == 31


# --- The slow presets' evidence script (W-899) ------------------------------------------------------------------------

SLOW_PRESETS_SCRIPT = "benchmarks/synthetic_scene_evidence.py"


def test_the_slow_preset_evidence_script_collects_exactly_its_moved_tests():
    """``--list`` exits 1 unless the script collects exactly its MOVED cases, each with its former node ID."""
    done = subprocess.run([sys.executable, str(REPOSITORY / SLOW_PRESETS_SCRIPT), "--list"],
                          cwd=REPOSITORY / "src/python",
                          env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    listing = json.loads(done.stdout)
    assert listing["complete"], listing
    presets = ["context-dense3d-slow-v1", "context-tissue2d-slow-v1", "context-culture-slow-v1"]
    expected = {**{f"src/python/test/test_synthetic_cellular_presets.py::{test}[{name}]": "K5" for name in presets
                   for test in ("test_gn7_slow_preset_reaches_its_floor", "test_gn5_lobes_on_the_slow_presets")},
                **{"src/python/test/test_synthetic_appearance_golden.py::"
                   f"test_saved_context_preset_outputs_equal_the_base_revision_pins[{name}]": "O1" for name in presets}}
    assert {case["origin_node_id"]: case["group"] for case in listing["cases"]} == expected
    assert len(listing["cases"]) == len(expected) == 9
    assert {case["node_id"].split("::", 1)[1].split("[", 1)[0] for case in listing["cases"]} == \
        script_tests(SLOW_PRESETS_SCRIPT)
