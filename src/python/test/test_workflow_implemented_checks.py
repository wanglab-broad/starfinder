"""The validation inventory names actual collected tests, a bounded script and human records.

A pytest row may name an evidence script under ``benchmarks/`` and its tests (W-898 moved G3 and some cases of X11 there):
the script defines those tests, and each script collects exactly its list of moved tests, each with the node ID it had in
the suite (``--list``, in a process of its own).
"""
from __future__ import annotations

import ast
import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from . import docs_pages

pytestmark = [pytest.mark.workflow, pytest.mark.contract]
pytest_plugins = [__package__ + ".workflow_inventory"]
ROOT = Path(__file__).resolve().parents[3]


#: The evidence scripts that hold tests moved out of the suite (W-898).
EVIDENCE_SCRIPTS = ("benchmarks/workflow_engine_evidence.py", "benchmarks/orchestration_golden_evidence.py")


@functools.cache
def _evidence_tests(script):
    """The test functions an evidence script under benchmarks/ defines at its top level (read, not imported)."""
    tree = ast.parse((ROOT / script).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def _ids(cell):
    match = re.fullmatch(r"([CNGXU])(\d+)(?: to [CNGXU](\d+))?", cell)
    assert match, cell
    prefix, start, end = match.groups()
    return {prefix + str(i) for i in range(int(start), int(end or start) + 1)}


def test_implemented_checks_name_collected_tests_scripts_and_human_records(request):
    from . import workflow_inventory
    text = docs_pages.design("validation/orchestration.md")
    table = docs_pages.table(text, "orchestration-validation-implemented-checks")
    rows = [[c.strip() for c in row.strip('|').split('|')] for row in table.splitlines()[2:]]
    expected = set()
    for cell in re.findall(r"^\| ([CNGXU]\d+) \|", text.split("(orchestration-validation-implemented-checks)=")[0], re.M):
        assert cell not in expected, cell
        expected.add(cell)
    actual = set()
    modules = set()
    for row in rows:
        ids = _ids(row[0])
        assert not ids & actual, ids & actual
        actual |= ids
        if row[2] not in ("script", "owner-run"):
            modules.update(re.findall(r"`(test_\w+\.py)`", row[1]))
    assert actual == expected
    # Q has already collected the whole suite before deselecting Extended/Slow.
    # Reuse that real collection; a file-only focused invocation collects just
    # the missing modules in a second interpreter, without executing their tests.
    collected = set(request.config.stash[workflow_inventory.COLLECTED])
    present = {Path(node.split('::', 1)[0]).name for node in collected}
    missing = modules - present
    if missing:
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-m", "",
                   *["test/" + module for module in sorted(missing)]]
        environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        done = subprocess.run(command, cwd=ROOT / "src/python", env=environment,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stdout + done.stderr
        collected.update(line.split('[', 1)[0] for line in done.stdout.splitlines()
                         if line.startswith('test/') and '::' in line)
    for check, names, selection, issue in rows:
        if selection == "script":
            paths = re.findall(r"`([^`]+)`", names)
            assert len(paths) == 2 and (ROOT / paths[0]).is_file(), paths
            output = Path(paths[1])
            assert output.as_posix() == "runs/W-765/20261102T160616Z-e8bba206/results/x9/result.json"
            # Existence belongs to the run manifest and results gate, never this host-independent audit.
        elif selection == "owner-run":
            assert issue == "TG4"
            assert re.findall(r"`([^`]+)`", names) == [f"results/uger/{check}.md"]
        else:
            segments = re.split(r"(?=`(?:test_\w+\.py|benchmarks/\w+\.py)`)", names)
            found = []
            for segment in segments:
                match = re.match(r"`(test_\w+\.py|benchmarks/\w+\.py)`", segment)
                if not match:
                    continue
                module = match.group(1)
                tests = re.findall(r"`(test_\w+)`", segment)
                assert tests, (check, module)
                for test in tests:
                    if module.startswith("benchmarks/"):
                        assert test in _evidence_tests(module), (check, module, test)
                        found.append(f"{module}::{test}")
                        continue
                    node = f"test/{module}::{test}"
                    assert node in collected, (check, node)
                    found.append(node)
            assert found, check


@pytest.mark.parametrize("script", EVIDENCE_SCRIPTS, ids=lambda script: Path(script).stem)
def test_each_evidence_script_collects_exactly_its_moved_tests(script):
    """``--list`` collects the script's tests with pytest and exits 1 unless they are exactly its list of moved tests
    (``MOVED``), each with the node ID it had in the suite; every test function the script defines is collected."""
    done = subprocess.run([sys.executable, str(ROOT / script), "--list"], cwd=ROOT / "src/python",
                          env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    listing = json.loads(done.stdout)
    assert listing["complete"] and listing["cases"], listing
    assert all(case["origin_node_id"].startswith("src/python/test/test_") for case in listing["cases"]), listing
    assert {case["node_id"].split("::", 1)[1].split("[", 1)[0] for case in listing["cases"]} == _evidence_tests(script)
