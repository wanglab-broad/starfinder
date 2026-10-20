#!/usr/bin/env python3
"""Compute the dependency closure of the learned tests and check that a check's trigger inputs cover it.

The learned checks run only when one of their trigger paths changes (docs/contributing.md, "Test
selections": learned tests run when a dependency of theirs changes). This script finds those
dependencies from the source, without running a test:

* the selection: each check's node IDs, collected with ``pytest --collect-only -m <marker> [-k <filter>]``
  in its own process, as the check itself would select them;
* the closure: the test files of the selected node IDs, ``test/conftest.py`` and ``test/__init__.py``,
  and every module of ``starfinder`` and of the test package that they import, directly or
  transitively. Imports are read from the syntax tree of each file, including imports inside
  functions (``Dataset.report`` imports ``starfinder.report`` when it is called) and imports in the
  code strings a test runs in a subprocess (``python -c``). Importing ``a.b.c`` also runs
  ``a/__init__.py`` and ``a/b/__init__.py``, so those are in the closure too. The data files,
  documentation pages and other repository files that a test module names by a string constant
  (``test/data/<name>``, ``docs/<page>.md``, ``workflow/schemas/config.schema.yaml``) are listed
  separately;
* a cross-check: each root test file is imported in a fresh process and the ``starfinder`` and test
  modules then loaded must all be in the static closure;
* with ``--checks``: the overlap of the checks' selections, and for each check the closure modules
  that none of its ``inputs`` match (``fnmatch``, as the runner matches changed paths) and the
  ``inputs`` that match no file of the worktree. ``--propose`` adds, per check, the inputs that keep
  the existing ones that still match a file and add a pattern for each closure file they miss.

Run from src/python::

    uv run python scripts/learned_closure.py --output <dir>
    uv run python scripts/learned_closure.py --output <dir> --checks <checks.json> --select 'pytest-learned*' --propose

``checks.json`` holds a list of checks, or an object whose ``checks`` or ``config.checks`` is that
list; each check has a ``name``, a pytest ``command`` with ``-m`` (and optionally ``-k``) and
``inputs``. Without ``--checks`` the one selection is ``-m learned``. The exit status is 1 when a
closure module is outside the static closure's cross-check, a closure file is not matched by its
check's inputs or an input matches no file, and 0 otherwise (with ``--propose``, of the proposal).
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/python
REPO_ROOT = PACKAGE_ROOT.parents[1]
TESTS = PACKAGE_ROOT / "test"
LOCAL = ("starfinder", "test")  # the top-level packages whose modules make up the closure
ALWAYS = ("test/conftest.py", "test/__init__.py")  # loaded by pytest for every test
ENVIRONMENT = ("src/python/pyproject.toml", "src/python/uv.lock")  # the locked environment
_IMPORT_TEXT = re.compile(r"(?:^|[;\s])(?:from\s+([\w.]+)\s+import\s+([\w\s,()]+)|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))")


# ============================================================================================ selection

class _Recorder:
    """Pytest plugin that writes the node IDs of the selected items, one per line."""

    def __init__(self, path: Path):
        self.path = path

    def pytest_collection_finish(self, session):
        self.path.write_text("".join(item.nodeid + "\n" for item in session.items))


def _collect_worker(marker: str, keyword: str, out: Path) -> int:
    import pytest

    arguments = ["test/", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", marker]
    if keyword:
        arguments += ["-k", keyword]
    return int(pytest.main(arguments, plugins=[_Recorder(out)]))


def collect(marker: str, keyword: str, out: Path) -> list[str]:
    """The node IDs ``pytest test/ --collect-only -m marker [-k keyword]`` selects, run from src/python."""
    status = subprocess.run([sys.executable, __file__, "--collect-worker", marker, keyword, str(out)],
                            cwd=PACKAGE_ROOT, stdout=subprocess.DEVNULL).returncode
    # 5: pytest found no item for the expression.
    if status not in (0, 5):
        raise RuntimeError(f"collection of -m {marker!r} -k {keyword!r} exited with {status}")
    return out.read_text().splitlines() if status == 0 and out.exists() else []


def selection_of(command: list[str]) -> tuple[str, str]:
    """The ``-m`` and ``-k`` values of a pytest command; the last of each wins, as in pytest."""
    marker = keyword = ""
    for flag, value in zip(command, command[1:]):
        if flag == "-m":
            marker = value
        elif flag == "-k":
            keyword = value
    return marker, keyword


# ============================================================================================ closure

def module_file(name: str) -> Path | None:
    """The source file of a module of the package root (``a.b`` is a/b.py or a/b/__init__.py)."""
    if name.split(".")[0] not in LOCAL:
        return None
    base = PACKAGE_ROOT.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _with_parents(name: str) -> list[str]:
    """``a.b.c`` imports a, a.b and a.b.c."""
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _resolve(current: str, is_package: bool, level: int, module: str | None) -> str:
    """The absolute name of a relative import ``from <level dots><module> import ...`` in module current."""
    if not level:
        return module or ""
    parts = current.split(".") if is_package else current.split(".")[:-1]
    parts = parts[:len(parts) - (level - 1)] if level > 1 else parts
    return ".".join(parts + ([module] if module else []))


def _code_strings(tree: ast.AST):
    """The text of each string in the tree that may be Python code with an import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "import" in node.value:
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in node.values)
            if "import" in text:
                yield text


def _string_imports(text: str) -> list[tuple[str, list[str]]]:
    """The ``(module, imported names)`` of the import statements in a code string."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        found = []
        for match in _IMPORT_TEXT.finditer(text):
            if match.group(1):
                found.append((match.group(1), [n for n in re.split(r"[\s,()]+", match.group(2)) if n]))
            else:
                found += [(name.strip(), []) for name in match.group(3).split(",")]
        return found
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.append((node.module, [alias.name for alias in node.names]))
    return found


def imports_of(path: Path) -> set[str]:
    """The modules of the package root that the file imports anywhere: at the top, in functions, in code strings."""
    current = module_name(path)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            wanted += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            wanted.append((_resolve(current, is_package, node.level, node.module), [a.name for a in node.names]))
    for text in _code_strings(tree):
        wanted += _string_imports(text)
    found = set()
    for module, names in wanted:
        if not module:
            continue
        # `from a.b import c` imports a.b and, when c is a submodule, a.b.c as well.
        for name in _with_parents(module) + [f"{module}.{n}" for n in names if n != "*"]:
            if module_file(name) is not None:
                found.add(name)
    return found


def closure(roots: list[Path]) -> dict[str, dict]:
    """Each module reached from the roots: {name: {"path": repo-relative path, "via": the first importer}}."""
    reached: dict[str, dict] = {}
    queue = []
    for root in roots:
        name = module_name(root)
        for parent in _with_parents(name):
            if parent not in reached and module_file(parent) is not None:
                reached[parent] = {"path": _repo_path(module_file(parent)), "via": "root" if parent == name
                                   else f"package of {name}"}
                queue.append(parent)
    while queue:
        name = queue.pop()
        for imported in sorted(imports_of(module_file(name))):
            if imported not in reached:
                reached[imported] = {"path": _repo_path(module_file(imported)), "via": name}
                queue.append(imported)
    return dict(sorted(reached.items()))


def named_files(modules: dict[str, dict]) -> dict[str, str]:
    """The data files and documentation pages that the test modules name by a string constant: {path: module}."""
    found = {}
    data = TESTS / "data"
    entries = {p.name for p in data.iterdir()} if data.is_dir() else set()
    pages = {p.name for p in (REPO_ROOT / "docs").glob("*.md")}
    for name, info in modules.items():
        if not name.startswith("test"):
            continue
        tree = ast.parse((REPO_ROOT / info["path"]).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
                value = node.value
                if value in entries:
                    found.setdefault(_repo_path(data / value), name)
                elif value in pages:
                    found.setdefault(f"docs/{value}", name)
                elif "/" in value and not value.startswith("/") and "\n" not in value:
                    # A path relative to the repository or the package root (workflow/schemas/...).
                    for base in (REPO_ROOT, PACKAGE_ROOT):
                        if (base / value).is_file():
                            found.setdefault(_repo_path(base / value), name)
                            break
    return dict(sorted(found.items()))


def _import_worker(module: str, out: Path) -> int:
    import importlib

    sys.path.insert(0, str(PACKAGE_ROOT))
    importlib.import_module(module)
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in LOCAL and module_file(name) is not None)
    out.write_text("".join(name + "\n" for name in loaded))
    return 0


def imported_at_load(module: str, out: Path) -> list[str]:
    """The package-root modules loaded when the module is imported in a fresh process."""
    subprocess.run([sys.executable, __file__, "--import-worker", module, str(out)], cwd=PACKAGE_ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return out.read_text().splitlines()


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _roots(nodeids: list[str]) -> list[Path]:
    files = {PACKAGE_ROOT / nodeid.split("::")[0] for nodeid in nodeids}
    return sorted(files | {PACKAGE_ROOT / path for path in ALWAYS})


# ============================================================================================ trigger inputs

def tracked_files() -> list[str]:
    """The worktree's files, repo-relative: tracked ones and untracked ones that are not ignored."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(path for path in out.splitlines() if (REPO_ROOT / path).exists())


def matching(path: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if fnmatch.fnmatch(path, pattern)]


def pattern_for(path: str) -> str:
    """The input that covers a closure file: its subpackage for a module of a starfinder subpackage, the
    entry of test/data for a data file, and the file itself otherwise."""
    parts = path.split("/")
    if parts[:3] == ["src", "python", "starfinder"] and len(parts) > 4:
        return "/".join(parts[:4]) + "/*"
    if parts[:4] == ["src", "python", "test", "data"] and len(parts) > 4:
        return "/".join(parts[:5]) + "*"
    return path


def audit(files: list[str], inputs: list[str], worktree: list[str]) -> dict:
    """The closure files no input matches and the inputs that match no file of the worktree."""
    return {"unmatched": [path for path in files if not matching(path, inputs)],
            "stale": [pattern for pattern in inputs if not any(fnmatch.fnmatch(p, pattern) for p in worktree)]}


def propose(files: list[str], inputs: list[str], worktree: list[str]) -> list[str]:
    """The existing inputs that still match a file, then a pattern for each closure file they miss."""
    kept = [pattern for pattern in inputs if any(fnmatch.fnmatch(p, pattern) for p in worktree)]
    for path in files:
        if not matching(path, kept):
            kept.append(pattern_for(path))
    return kept


# ============================================================================================ main

def _load_checks(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("checks") or data.get("config", {}).get("checks")
    if not isinstance(data, list):
        raise SystemExit(f"{path}: no list of checks")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="directory for the node IDs and the reports")
    parser.add_argument("--checks", type=Path, help="JSON file with the checks (read only)")
    parser.add_argument("--select", default="pytest-learned*", help="fnmatch pattern of the check names to audit")
    parser.add_argument("--propose", action="store_true", help="write the proposed inputs of each check")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    (out / "selections").mkdir(parents=True, exist_ok=True)

    checks = [{"name": "learned", "command": ["pytest", "-m", "learned"], "inputs": []}]
    if args.checks:
        checks = [c for c in _load_checks(args.checks) if fnmatch.fnmatch(c["name"], args.select)]
    learned = collect("learned", "", out / "selections" / "learned-all.txt")
    print(f"{'-m learned':45s} {len(learned):5d}")

    selections, closures = {}, {}
    for check in checks:
        marker, keyword = selection_of(check["command"])
        ids = collect(marker, keyword, out / "selections" / f"{check['name']}.txt")
        selections[check["name"]] = {"marker": marker, "keyword": keyword, "count": len(ids),
                                     "files": sorted({i.split("::")[0] for i in ids}), "nodeids": ids}
        closures[check["name"]] = closure(_roots(ids))
        print(f"{check['name']:45s} {len(ids):5d}  -m {marker!r}" + (f" -k {keyword!r}" if keyword else ""))

    # The union over the audited checks, and its cross-check against the modules loaded at import.
    union: dict[str, dict] = {}
    for modules in closures.values():
        for name, info in modules.items():
            union.setdefault(name, info)
    roots = sorted({path for s in selections.values() for path in s["files"]})
    loaded = {}
    for root in roots:
        name = module_name(PACKAGE_ROOT / root)
        loaded[name] = imported_at_load(name, out / "selections" / "loaded.tmp")
    (out / "selections" / "loaded.tmp").unlink(missing_ok=True)
    missed = sorted({m for names in loaded.values() for m in names} - set(union))
    files = named_files(union)
    report = {
        "package_root": _repo_path(PACKAGE_ROOT),
        "roots": roots + list(ALWAYS),
        "modules": len(union),
        "closure": union,
        "named_files": files,
        "per_check": {name: sorted(info["path"] for info in modules.values()) for name, modules in closures.items()},
        "per_check_named_files": {name: sorted(named_files(modules)) for name, modules in closures.items()},
        "loaded_at_import_not_in_closure": missed,
    }
    (out / "closure.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "closure.txt").write_text("".join(f"{info['path']}\t{name}\t{info['via']}\n" for name, info in union.items()))
    print(f"closure: {len(union)} modules, {len(files)} named data files and pages; "
          f"loaded at import but not in the closure: {len(missed)}")

    # Which node IDs two checks share, and whether the checks together select each learned test once.
    names = list(selections)
    sets = {name: set(selections[name]["nodeids"]) for name in names}
    overlap = {f"{a} & {b}": sorted(sets[a] & sets[b]) for a, b in combinations(names, 2)}
    counts = defaultdict(int)
    for name in names:
        for nodeid in sets[name]:
            counts[nodeid] += 1
    overlaps = {
        "checks": {name: {k: v for k, v in selections[name].items() if k != "nodeids"} for name in names},
        "pairs": {pair: {"count": len(ids), "nodeids": ids} for pair, ids in overlap.items()},
        "learned_not_selected": sorted(set(learned) - set(counts)),
        "selected_not_learned": sorted(set(counts) - set(learned)),
        "selected_twice_or_more": sorted(i for i, n in counts.items() if n > 1),
    }
    (out / "overlap.json").write_text(json.dumps(overlaps, indent=1) + "\n")
    lines = ["| check | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(str(len(sets[a])) if a == b else str(len(sets[a] & sets[b]))
                                              for b in names) + " |")
    (out / "overlap.md").write_text("Node IDs selected by both checks (the diagonal: by the check).\n\n"
                                    + "\n".join(lines) + "\n")
    print(f"pairs sharing a node ID: {sum(1 for ids in overlap.values() if ids)}; learned tests in no check: "
          f"{len(overlaps['learned_not_selected'])}; in two or more: {len(overlaps['selected_twice_or_more'])}")

    if not args.checks:
        return 1 if missed else 0

    worktree = tracked_files()
    ok = not missed
    result = {"current": {}, "proposed": {}}
    for check in checks:
        name = check["name"]
        needed = report["per_check"][name] + report["per_check_named_files"][name] + list(ENVIRONMENT)
        result["current"][name] = audit(needed, check.get("inputs", []), worktree)
        if args.propose:
            inputs = propose(needed, check.get("inputs", []), worktree)
            result["proposed"][name] = {"inputs": inputs, **audit(needed, inputs, worktree),
                                        "added": [p for p in inputs if p not in check.get("inputs", [])],
                                        "dropped": [p for p in check.get("inputs", []) if p not in inputs],
                                        "matches": {path: matching(path, inputs) for path in needed}}
            ok = ok and not (result["proposed"][name]["unmatched"] or result["proposed"][name]["stale"])
        else:
            ok = ok and not (result["current"][name]["unmatched"] or result["current"][name]["stale"])
        current = result["current"][name]
        print(f"{name:45s} current inputs: {len(current['unmatched'])} closure files unmatched, "
              f"{len(current['stale'])} stale" + (f"; proposed: {len(result['proposed'][name]['unmatched'])} "
                                                  f"unmatched, {len(result['proposed'][name]['stale'])} stale"
                                                  if args.propose else ""))
    (out / "inputs.json").write_text(json.dumps(result, indent=1) + "\n")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--collect-worker":
        sys.exit(_collect_worker(sys.argv[2], sys.argv[3], Path(sys.argv[4])))
    if len(sys.argv) == 4 and sys.argv[1] == "--import-worker":
        sys.exit(_import_worker(sys.argv[2], Path(sys.argv[3])))
    sys.exit(main())
