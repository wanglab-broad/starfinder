"""The table "Implemented checks" of docs/synthetic-real-appearance-validation.md, and GA25 (W-520).

``test_every_ga_row_names_collected_tests`` keeps the table true: it reads the rows GA1 to GA27 of the table
"Generator checks" and checks that "Implemented checks" has exactly those rows, in that order; that each names at
least one test pytest collects; that none of them is an expected failure or skipped by a mark; that each carries the
subsystem marker ``synthetic`` and the kind marker its GA row names; and that the tier of the line matches the tests'
``slow`` markers. ``test_ga25_existing_tests_unchanged`` is GA25: the files GN25 pins and the test modules and helpers
of W-483 to W-486, with their content at 8bf9a5b (the base revision of the appearance implementation batch), pinned by
SHA-256 (line endings read as LF); the session fixture ``small_dataset`` of conftest.py by GN25's pins of its source.
"""
import hashlib
from pathlib import Path
import re

import pytest

from .test_synthetic_cellular_checks import CONFTEST_PINS, PINS, conftest_sources

pytestmark = [pytest.mark.synthetic]

TEST = Path(__file__).resolve().parent
DESIGN = TEST.parents[2] / "docs" / "synthetic-real-appearance-validation.md"
SECTION = "## Implemented checks\n"
TIERS = {"routine": (False,), "slow": (True,), "routine and slow": (False, True)}
KINDS = {"contract", "validation", "golden"}

# GA25: the test modules and helpers of W-483 to W-486 at 8bf9a5b. The files GN25 pins are PINS of
# test_synthetic_cellular_checks.py (their content at 7d6c417, the same at 8bf9a5b), a module pinned here.
CELLULAR_PINS = {
    "cellular_checks.py": "a767a5bad0354597a4d3acae2f5d9ce4b9230486d8f53e2cd88a8bb854ad6fa7",
    "cellular_measures.py": "5041fdb0a39f502f6f3138b29951c61f29c1e96881870e044eceac6f69a4bf15",
    "cellular_oracle.py": "92749ac589ddfe104d7e5fedfbceeffdae39f445242aff3d4a883d35ec5026ae",
    "cellular_scenes.py": "89b41b4b380ceb81fae2bb251ddac6a4c73c9598993254d4ad16260272e12f07",
    # Changed on 2026-10-19 with GN25's pin of test_synthetic_sample_golden.py (GN1's amendment).
    "test_synthetic_cellular_checks.py": "0d6d9a36d1e20977721cd677508e78fde7bfb0cbeb6ca18a87ac3fc1466baafa",
    "test_synthetic_cellular_contract.py": "8d1ebb0a2dae8be379014fc976961560ff04716c7883651551e237b6d64441af",
    "test_synthetic_cellular_presets.py": "9d73cff301b6043dd839379d27f67cbf605c9b85c92cde8717086c32353b70bf",
    "test_synthetic_cellular_presets_contract.py": "3e98639567d3430c46f09b08115b9353602b49839fc12088cd5d3eb5a7fc303f",
    "test_synthetic_cellular_routine.py": "c2689bbc8359e1a4c86732a7236d444292a9aebfa7ea593a862a24f552aeae30",
    "test_synthetic_cellular_scene.py": "20041f1c64c84029190470cbf82ae17f80977e0b65638e28b6b7781e6467746d",
    "test_synthetic_cellular_stains.py": "ddf1dbdb4af9db81aa79665f49fdcac7ab9dc67d00ce5a25fbe471ac1daf72e5",
    "test_synthetic_cellular_stains_contract.py": "1b8b481e9c5525cdcecdc25625737de3d8f99df5a6cca1395e495503ecbdfca2",
}


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.validation
def test_ga25_existing_tests_unchanged():
    pins = dict(PINS, **CELLULAR_PINS)
    assert len(pins) == len(PINS) + len(CELLULAR_PINS)
    changed = [name for name, pin in sorted(pins.items()) if digest((TEST / name).read_bytes()) != pin]
    sources = conftest_sources()
    assert sorted(sources) == sorted(CONFTEST_PINS)
    changed += [f"conftest.py::{name}" for name, pin in CONFTEST_PINS.items()
                if hashlib.sha256(sources[name].encode("utf-8")).hexdigest() != pin]
    assert not changed, f"GA25: files that must stay as at 8bf9a5b were edited: {changed}"
    # Every test module and helper of W-483 to W-486 is pinned: the cellular_ helpers other than the appearance
    # scene's (cellular_appearance_*.py) and every test_synthetic_cellular_*.py.
    present = {p.name for p in TEST.glob("test_synthetic_cellular_*.py")}
    present |= {p.name for p in TEST.glob("cellular_*.py") if not p.name.startswith("cellular_appearance_")}
    assert present == set(CELLULAR_PINS), sorted(present ^ set(CELLULAR_PINS))
    print(f"GA25: {len(pins)} files and {len(CONFTEST_PINS)} conftest.py sources as at 8bf9a5b")


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
    assert list(expected) == [f"GA{k}" for k in range(1, 28)]
    assert list(rows) == list(expected)
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
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
