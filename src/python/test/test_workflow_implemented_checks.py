"""The validation inventory names actual collected tests, a bounded script and human records."""
from __future__ import annotations

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
            segments = re.split(r"(?=`test_\w+\.py`)", names)
            found = []
            for segment in segments:
                match = re.match(r"`(test_\w+\.py)`", segment)
                if not match:
                    continue
                module = match.group(1)
                tests = re.findall(r"`(test_\w+)`", segment)
                assert tests, (check, module)
                for test in tests:
                    node = f"test/{module}::{test}"
                    assert node in collected, (check, node)
                    found.append(node)
            assert found, check
