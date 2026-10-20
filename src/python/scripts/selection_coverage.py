#!/usr/bin/env python3
"""Collect the node IDs of each test selection and check that the selections cover the suite.

The selections are defined in docs/contributing.md, "Test selections" (W-530). Each one is
collected with ``pytest --collect-only -m <expression>`` in its own process, so the node IDs are
those pytest itself selects; no test runs. The report states:

* the overlap of the quick gate Q and the remaining fast validation V (must be empty), and
  whether Q and V together equal the whole fast tier R;
* whether R, slow, learned and extended together hold each collected node ID exactly once;
* whether the slow subsystem checks together equal the slow selection (a test with two
  subsystems may be in two of them; that overlap is reported, not an error).

Run from src/python::

    uv run python scripts/selection_coverage.py --output <dir>

The exit status is 0 when both coverage conditions hold and 1 otherwise.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

FAST = "not (slow or learned or extended)"

# The pytest selections and their marker expressions; docs/contributing.md lists the same.
SELECTIONS = {
    "Q": f"{FAST} and (contract or not validation)",
    "V": f"validation and not contract and {FAST}",
    "R": FAST,
    "slow": "slow and not (learned or extended)",
    "learned": "learned",
    "extended": "extended and not learned",
}

# The slow selection is run per subsystem group, as the controller's existing slow checks are.
SLOW = SELECTIONS["slow"]
SLOW_GROUPS = {
    "slow-benchmark": f"{SLOW} and benchmark",
    "slow-synthetic": f"{SLOW} and synthetic and not benchmark",
    "slow-registration": f"{SLOW} and registration",
    "slow-spot-finding": f"{SLOW} and spot_finding",
    "slow-other": f"{SLOW} and not (benchmark or synthetic or registration or spot_finding)",
}

# R, slow, learned and extended partition the collected suite.
PARTITION = ("R", "slow", "learned", "extended")


def _defined_under(function, root: Path) -> bool:
    try:
        return Path(inspect.getsourcefile(function)).resolve().is_relative_to(root)
    except (TypeError, OSError):
        return False


class _Recorder:
    """Pytest plugin that writes the selected items as JSON lines: the node ID, the marker names and the
    fixtures wider than the function (module, class, package or session) that the test suite itself defines."""

    def __init__(self, path: Path, root: Path):
        self.path, self.root = path, root

    def pytest_collection_finish(self, session):
        with open(self.path, "w") as f:
            for item in session.items:
                defs = getattr(getattr(item, "_fixtureinfo", None), "name2fixturedefs", {})
                shared = sorted(name for name, chain in defs.items() if chain and chain[-1].scope != "function"
                                and _defined_under(chain[-1].func, self.root))
                f.write(json.dumps({"nodeid": item.nodeid,
                                    "markers": sorted({m.name for m in item.iter_markers()}),
                                    "shared_fixtures": shared}) + "\n")


def _collect_worker(expression: str, out: Path, tests: str) -> int:
    import pytest

    return int(pytest.main([tests, "--collect-only", "-q", "-p", "no:cacheprovider", "-m", expression],
                           plugins=[_Recorder(out, Path(tests).resolve())]))


def collect(expression: str, out: Path, tests: str = "test/") -> list[dict]:
    """The items ``pytest --collect-only -m expression`` selects, each {nodeid, markers, shared_fixtures}.

    ``expression`` "" selects every collected item (it replaces the ``-m 'not extended'`` of addopts).
    """
    status = subprocess.run([sys.executable, __file__, "--collect-worker", expression, str(out), tests],
                            stdout=subprocess.DEVNULL).returncode
    # 5: pytest found no item for the expression.
    if status not in (0, 5):
        raise RuntimeError(f"collection of {expression!r} exited with {status}")
    if status == 5 or not out.exists():
        return []
    return [json.loads(line) for line in out.read_text().splitlines()]


def coverage(ids: dict[str, list[str]], everything: list[str]) -> dict:
    """The coverage report of the collected node IDs of each selection (see the module docstring)."""
    sets = {name: set(values) for name, values in ids.items()}
    q_and_v = sorted(sets["Q"] & sets["V"])
    counts = Counter(i for name in PARTITION for i in ids[name])
    collected = set(everything)
    missing = sorted(collected - set(counts))
    repeated = sorted(i for i, n in counts.items() if n > 1)
    outside = sorted(set(counts) - collected)
    slow_counts = Counter(i for name in SLOW_GROUPS for i in ids.get(name, []))
    report = {
        "collected": len(everything),
        "collected_unique": len(collected),
        "counts": {name: len(values) for name, values in ids.items()},
        "q_and_v_overlap": q_and_v,
        "q_union_v_equals_r": (sets["Q"] | sets["V"]) == sets["R"],
        "partition": list(PARTITION),
        "partition_missing": missing,
        "partition_repeated": repeated,
        "partition_outside_collection": outside,
        "partition_each_once": not (missing or repeated or outside) and len(collected) == len(everything),
    }
    if all(name in ids for name in SLOW_GROUPS):
        report["slow_groups_union_equals_slow"] = set(slow_counts) == sets["slow"]
        report["slow_groups_in_two_or_more"] = sorted(i for i, n in slow_counts.items() if n > 1)
    report["ok"] = not q_and_v and report["q_union_v_equals_r"] and report["partition_each_once"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the node-ID lists, markers.jsonl and coverage.json")
    parser.add_argument("--tests", default="test/", help="test path to collect (default test/)")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    selections = {"all": "", **SELECTIONS, **SLOW_GROUPS}
    ids = {}
    for name, expression in selections.items():
        items = collect(expression, args.output / f"{name}.jsonl", args.tests)
        ids[name] = [item["nodeid"] for item in items]
        (args.output / f"{name}.txt").write_text("".join(i + "\n" for i in ids[name]))
        if name != "all":
            (args.output / f"{name}.jsonl").unlink(missing_ok=True)
        print(f"{name:20s} {len(ids[name]):6d}  -m {expression!r}")
    (args.output / "all.jsonl").replace(args.output / "markers.jsonl")

    everything = ids.pop("all")
    report = {"selections": selections, **coverage(ids, everything)}
    (args.output / "coverage.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"collected {report['collected']} node IDs ({report['collected_unique']} distinct)")
    print(f"Q and V share {len(report['q_and_v_overlap'])} node IDs; Q | V == R: {report['q_union_v_equals_r']}")
    print(f"R, slow, learned and extended hold each collected node ID once: {report['partition_each_once']} "
          f"(missing {len(report['partition_missing'])}, repeated {len(report['partition_repeated'])}, "
          f"outside the collection {len(report['partition_outside_collection'])})")
    if "slow_groups_union_equals_slow" in report:
        print(f"slow subsystem checks together equal slow: {report['slow_groups_union_equals_slow']} "
              f"({len(report['slow_groups_in_two_or_more'])} node IDs in two or more groups)")
    print("OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--collect-worker":
        sys.exit(_collect_worker(sys.argv[2], Path(sys.argv[3]), sys.argv[4]))
    sys.exit(main())
