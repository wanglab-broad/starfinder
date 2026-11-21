#!/usr/bin/env python3
"""Write, for each pytest check of the runner, the test files it runs and the modules they import.

The runner starts a check when a changed path matches one of the check's inputs. Inputs named by glob
(``src/python/test/workflow_*.py``) rerun a check for a helper its tests never import, so W-893 (H3)
names each check's inputs by its import closure instead. This script computes that closure:

* the selection: each check of ``CHECKS`` is collected with ``pytest --collect-only`` and the check's
  own arguments (paths, ``--ignore``, ``-m``, ``-k``) in its own process, from src/python, so the test
  files are those of the node IDs pytest itself selects; no test runs;
* the closure: the test files, ``test/conftest.py`` and ``test/__init__.py``, and every module of
  ``starfinder`` and of the test package they import, directly or through another module, read from the
  syntax tree of each file by ``scripts/learned_closure.py`` (imports at the top, in functions and in code
  strings, the packages' ``__init__.py`` files, and files loaded by their path);
* the named files: the data files and documentation pages the test modules name by a string constant;
* the loads by a path that cannot be evaluated (``runpy.run_path`` of a command read from a page), named by
  file and line, so that the check's inputs can name what they load by hand.

``CHECKS`` holds the runner's pytest checks as they are after W-893's splits S1 (the fast tier in two
checks by subsystem marker) and S2 (``pytest-extended-recovery`` in two checks by test). The result is
``test/check_inputs.json``: per check its arguments, ``test_files``, ``modules`` (the closure without the
test files), ``named_files`` and ``unresolved_loads``, all repository paths; and ``uncollected``, the test
files that no check selects. The coordinating session copies it into the runner's configuration.
``test/test_check_inputs.py`` keeps the modules, named files and unresolved loads of each check equal to
the import graph of its test files; a change to which tests a check selects (a new test file, a moved
test, a changed marker) needs this script again.

Run from src/python::

    uv run python scripts/check_inputs.py            # writes test/check_inputs.json
    uv run python scripts/check_inputs.py --check    # compares with it; exit status 1 on a difference

A check whose collection fails stops the script with an error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learned_closure as lc  # noqa: E402

PACKAGE_ROOT = lc.PACKAGE_ROOT  # src/python
OUTPUT = PACKAGE_ROOT / "test" / "check_inputs.json"
FAST = "not (slow or learned or extended)"
SLOW = "slow and not (learned or extended)"
# S1: the fast tier in two checks of about equal time (12.5 and 12.6 minutes in the fast tier of 21d7d82).
FAST_FIRST = "report or segmentation or synthetic or workflow"
# S2: pytest-extended-recovery in two checks by test (about 9 and 10 minutes in W-898's validation).
RECOVERY_FIRST = ("test_x7_sigkill_partial_task_recovers_with_only_its_downstream or "
                  "test_x11_stale_results_rerun_only_readers_and_downstream")
RECOVERY = "test/test_workflow_recovery.py"
ENGINES = ("test/test_workflow_engines_e2e.py", "test/test_workflow_engines_review.py",
           "test/test_workflow_nextflow_cache.py")

# The runner's pytest checks: the arguments that follow `pytest` in each check's command.
CHECKS = {
    "pytest-quick": ["test/", "-m", f"{FAST} and (contract or not validation)"],
    "pytest-ci-environment": ["test/"],
    "pytest-fast-tier-1": ["test/", "-m", f"{FAST} and ({FAST_FIRST})"],
    "pytest-fast-tier-2": ["test/", "-m", f"{FAST} and not ({FAST_FIRST})"],
    "pytest-slow-benchmark": ["test/", "-m", f"{SLOW} and benchmark"],
    "pytest-slow-synthetic": ["test/", "-m", f"{SLOW} and synthetic and not benchmark"],
    "pytest-slow-registration": ["test/", "-m", f"{SLOW} and registration"],
    "pytest-slow-spot-finding": ["test/", "-m", f"{SLOW} and spot_finding"],
    "pytest-slow-other": ["test/", "-m", f"{SLOW} and not (benchmark or synthetic or registration or spot_finding)"],
    "pytest-learned-slow-piscis-20230905": ["test/", "-m", "learned and slow", "-k", "20230905"],
    "pytest-learned-slow-piscis-20251212": ["test/", "-m", "learned and slow", "-k", "20251212"],
    "pytest-learned-slow-other": ["test/", "-m", "learned and slow and not integration",
                                  "-k", "not 20230905 and not 20251212"],
    "pytest-learned-slow-chain-culture": ["test/", "-m", "learned and slow and integration", "-k", "test_cu"],
    "pytest-learned-slow-chain-other": ["test/", "-m", "learned and slow and integration", "-k", "not test_cu"],
    "pytest-learned-fast": ["test/", "-m", "learned and not slow"],
    "pytest-extended-recovery-1": [RECOVERY, "-m", "extended and not learned", "-k", RECOVERY_FIRST],
    "pytest-extended-recovery-2": [RECOVERY, "-m", "extended and not learned", "-k", f"not ({RECOVERY_FIRST})"],
    "pytest-extended-other": ["test/", f"--ignore={RECOVERY}", "-m", "extended and not learned"],
    "pytest-nextflow-recovery": [RECOVERY, "-m", "nextflow"],
    "pytest-nextflow-engines": [*ENGINES, "-m", "nextflow"],
    "pytest-nextflow-other": ["test/", f"--ignore={RECOVERY}", *(f"--ignore={path}" for path in ENGINES),
                              "-m", "nextflow"],
}


class _Recorder:
    """Pytest plugin that writes the node IDs of the selected items, one per line."""

    def __init__(self, path: Path):
        self.path = path

    def pytest_collection_finish(self, session):
        self.path.write_text("".join(item.nodeid + "\n" for item in session.items))


def _collect_worker(out: Path, arguments: list[str]) -> int:
    import pytest

    return int(pytest.main([*arguments, "--collect-only", "-q", "-p", "no:cacheprovider"], plugins=[_Recorder(out)]))


def collect(arguments: list[str]) -> list[str]:
    """The node IDs ``pytest <arguments> --collect-only`` selects, run from src/python in its own process."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "nodeids.txt"
        status = subprocess.run([sys.executable, __file__, "--collect-worker", str(out), *arguments],
                                cwd=PACKAGE_ROOT, stdout=subprocess.DEVNULL).returncode
        # 5: pytest found no item for the selection.
        if status not in (0, 5):
            raise RuntimeError(f"collection of {arguments} exited with {status}")
        return out.read_text().splitlines() if status == 0 and out.exists() else []


def repo_path(path: str) -> str:
    """The repository path of a path under src/python (``test/test_a.py`` is ``src/python/test/test_a.py``)."""
    return lc._repo_path(PACKAGE_ROOT / path)


class ImportGraph:
    """The modules each file imports, read once per file (``learned_closure.imports_of`` and ``path_loads``)."""

    def __init__(self):
        self.edges: dict[str, list[str]] = {}
        self.unresolved: dict[str, list[str]] = {}
        self.named: dict[str, dict[str, str]] = {}

    def imported(self, key: str) -> list[str]:
        """The closure keys a module (by name) or a file loaded by path (by repository path) imports or loads."""
        if key not in self.edges:
            source = lc._source(key)
            found = sorted(lc.imports_of(source, None if "/" in key else key))
            # A load by path is a call of one of the loaders by name: a file that never spells one has none.
            text = source.read_text()
            loads = lc.path_loads(source) if any(name in text for name in lc.LOADERS) else []
            for load in loads:
                if not load["files"]:
                    where = f"{lc._repo_path(source)}:{load['line']} {load['call']}"
                    self.unresolved.setdefault(key, []).append(where)
                found += [loaded for loaded in load["files"] if loaded not in found]
            self.edges[key] = found
        return self.edges[key]

    def named_files(self, key: str, path: str) -> dict[str, str]:
        """``learned_closure.named_files`` of one module, read once."""
        if key not in self.named:
            self.named[key] = lc.named_files({key: {"path": path}})
        return self.named[key]

    def closure(self, roots: list[Path]) -> dict[str, str]:
        """{key: repository path} of every module reached from the root files, the roots and their packages
        included, as ``learned_closure.closure`` reaches them."""
        reached: dict[str, str] = {}
        queue = []
        for root in roots:
            for parent in lc._with_parents(lc.module_name(root)):
                if parent not in reached and lc.module_file(parent) is not None:
                    reached[parent] = lc._repo_path(lc.module_file(parent))
                    queue.append(parent)
        while queue:
            for key in self.imported(queue.pop()):
                if key not in reached:
                    reached[key] = key if "/" in key else lc._repo_path(lc.module_file(key))
                    queue.append(key)
        return reached


def inputs_of(test_files: list[str], graph: ImportGraph) -> dict:
    """The modules (the closure without the test files themselves), the named files and the loads by a path that
    cannot be evaluated (``file:line call``) of a check's test files, given as repository paths."""
    roots = sorted({PACKAGE_ROOT / Path(path).relative_to("src/python") for path in test_files}
                   | {PACKAGE_ROOT / path for path in lc.ALWAYS})
    reached = graph.closure(roots) if test_files else {}
    named = {name for key, path in reached.items() for name in graph.named_files(key, path)}
    return {"modules": sorted(set(reached.values()) - set(test_files)),
            "named_files": sorted(named),
            "unresolved_loads": sorted(load for key in reached for load in graph.unresolved.get(key, []))}


def test_files_on_disk() -> list[str]:
    return sorted(lc._repo_path(path) for path in (PACKAGE_ROOT / "test").glob("test_*.py"))


def compute() -> dict:
    graph = ImportGraph()
    checks = {}
    for name, arguments in CHECKS.items():
        nodeids = collect(arguments)
        test_files = sorted({repo_path(nodeid.split("::")[0]) for nodeid in nodeids})
        checks[name] = {"arguments": arguments, "test_files": test_files, **inputs_of(test_files, graph)}
        print(f"{name:40s} {len(nodeids):5d} tests in {len(test_files):3d} files, "
              f"{len(checks[name]['modules']):3d} modules", file=sys.stderr)
    selected = {path for check in checks.values() for path in check["test_files"]}
    return {
        "about": "Generated by src/python/scripts/check_inputs.py (W-893 H3, W-900); do not edit. Per runner check: "
                 "its pytest arguments, the test files of the node IDs it selects, the modules they import (their "
                 "import closure without the test files), the files they name and the loads by a path that cannot be "
                 "evaluated (file:line call). Paths are repository paths.",
        "always": [lc._repo_path(PACKAGE_ROOT / path) for path in lc.ALWAYS],
        "checks": checks,
        "uncollected": [path for path in test_files_on_disk() if path not in selected],
    }


def text(result: dict) -> str:
    return json.dumps(result, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"file to write (default {OUTPUT.name} in test/)")
    parser.add_argument("--check", action="store_true", help="compare with --output instead of writing it")
    args = parser.parse_args(argv)
    result = text(compute())
    if args.check:
        same = args.output.is_file() and args.output.read_text() == result
        print(f"{args.output}: {'equal to' if same else 'differs from'} the import closure of the checks")
        return 0 if same else 1
    args.output.write_text(result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect-worker":
        sys.exit(_collect_worker(Path(sys.argv[2]), sys.argv[3:]))
    sys.exit(main())
