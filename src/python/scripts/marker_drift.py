#!/usr/bin/env python3
"""Report marker drift from a pytest ``--durations=0`` log (W-298 item 1, W-530).

Lists the tests that ran without ``slow`` and took 5 s or more (setup, call and teardown
together), and the ``slow`` tests that took less than 2 s. A ``slow`` test under 2 s that
requests a fixture wider than the function (module, class, package or session scope) is left
out of the second list and reported apart: it is ``slow`` because it shares a cached result
whose cost its first user pays (docs/contributing.md, "Test markers", rule 5). Skipped tests
are not counted. A one-time import is not separated here: a listed test is a candidate that a
measurement with the import done beforehand confirms or clears.

Run from src/python::

    uv run python scripts/marker_drift.py <durations.log> [--markers markers.jsonl] [--json out.json]

The markers come from ``--markers`` (the ``markers.jsonl`` that scripts/selection_coverage.py
writes) or, without it, from a collection of the whole suite (no test runs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SLOW_AT = 5.0
FAST_UNDER = 2.0

DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s (setup|call|teardown)\s+(\S.*?)\s*$")
OUTCOME = re.compile(r"^(\S+::\S.*?) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


def parse_log(text: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """({node ID: {phase: seconds}}, {node ID: outcome}) of a ``--durations=0`` log (outcomes from ``-v`` lines)."""
    durations: dict[str, dict[str, float]] = defaultdict(dict)
    outcomes: dict[str, str] = {}
    for line in text.splitlines():
        match = DURATION.match(line)
        if match:
            durations[match.group(3)][match.group(2)] = float(match.group(1))
            continue
        match = OUTCOME.match(line)
        if match:
            outcomes[match.group(1)] = match.group(2)
    return dict(durations), outcomes


def drift(durations: dict[str, dict[str, float]], outcomes: dict[str, str], items: dict[str, dict],
          slow_at: float = SLOW_AT, fast_under: float = FAST_UNDER) -> dict:
    """The drift report; ``items`` maps a node ID to {"markers": [...], "shared_fixtures": [...]}."""
    ran = {n for n in set(durations) | set(outcomes) if outcomes.get(n) != "SKIPPED"}
    report = {"slow_at": slow_at, "fast_under": fast_under, "tests": len(ran),
              "unmarked_at_or_over": [], "slow_under": [], "slow_under_shared_fixture": [], "not_collected": []}
    for node in sorted(ran):
        if node not in items:
            report["not_collected"].append(node)
            continue
        phases = durations.get(node, {})
        row = {"nodeid": node, "seconds": round(sum(phases.values()), 2), "phases": phases}
        slow = "slow" in items[node]["markers"]
        if not slow and row["seconds"] >= slow_at:
            report["unmarked_at_or_over"].append(row)
        elif slow and row["seconds"] < fast_under:
            shared = items[node].get("shared_fixtures", [])
            if shared:
                report["slow_under_shared_fixture"].append(dict(row, shared_fixtures=shared))
            else:
                report["slow_under"].append(row)
    for name in ("unmarked_at_or_over", "slow_under", "slow_under_shared_fixture"):
        report[name].sort(key=lambda r: (-r["seconds"], r["nodeid"]))
    return report


def text(report: dict) -> str:
    """The report as plain text."""
    lines = [f"{report['tests']} tests ran (skipped tests left out)", "",
             f"Without slow, {report['slow_at']:g} s or more: {len(report['unmarked_at_or_over'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}  {r['phases']}" for r in report["unmarked_at_or_over"]]
    lines += ["", f"slow, under {report['fast_under']:g} s: {len(report['slow_under'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}" for r in report["slow_under"]]
    lines += ["", f"Left out (slow, under {report['fast_under']:g} s, sharing a wider-scoped fixture): "
              f"{len(report['slow_under_shared_fixture'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}  {', '.join(r['shared_fixtures'])}"
              for r in report["slow_under_shared_fixture"]]
    if report["not_collected"]:
        lines += ["", f"In the log but not collected now: {len(report['not_collected'])}"]
        lines += [f"  {n}" for n in report["not_collected"]]
    return "\n".join(lines) + "\n"


def collected_items(markers: Path | None) -> dict[str, dict]:
    """{node ID: item} from a markers.jsonl, or from a collection of the whole suite."""
    if markers is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from selection_coverage import collect

        with tempfile.TemporaryDirectory() as directory:
            rows = collect("", Path(directory) / "all.jsonl")
    else:
        rows = [json.loads(line) for line in markers.read_text().splitlines() if line.strip()]
    return {row["nodeid"]: row for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="a pytest log written with -v --durations=0")
    parser.add_argument("--markers", type=Path, help="markers.jsonl of scripts/selection_coverage.py")
    parser.add_argument("--json", type=Path, help="also write the report as JSON here")
    args = parser.parse_args(argv)
    durations, outcomes = parse_log(args.log.read_text(errors="replace"))
    report = drift(durations, outcomes, collected_items(args.markers))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
