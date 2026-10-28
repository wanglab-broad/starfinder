#!/usr/bin/env python3
"""Report marker drift from a pytest ``--durations=0`` log (W-298 item 1, W-530).

Lists the tests that ran without ``slow`` and took 5 s or more (setup, call and teardown
together), and the ``slow`` tests that took less than 2 s. A ``slow`` test under 2 s is left
out of the second list, and reported apart with the entry's name, only when an entry of
``EXEMPTIONS`` covers it: the entry names one cached source (a wider-scoped fixture or a cached
helper that is not a fixture), the tests that share it, and a saved measurement of the cost its
first user pays, which makes the others ``slow`` (docs/contributing.md, "Test markers").
A wider-scoped fixture alone exempts nothing. Skipped tests are not counted. A one-time import is
not separated here: a listed test is a candidate that a measurement with the import done
beforehand confirms or clears.

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

# The cached sources that make tests ``slow`` although their own run is short. Each entry covers the tests it names
# ("<file>::<function>", every parametrization) and gives the measured cost its first user pays.
EXEMPTIONS = (
    {"name": "app_routine",
     "source": "test/cellular_appearance_scenes.py::app_routine (functools.cache, not a fixture)",
     "tests": ["test/test_synthetic_appearance_presets.py::test_ga8_routine_labels_equal_the_oracle_on_every_grid",
               "test/test_synthetic_appearance_presets.py::test_ga9_routine_ownership_overlap_and_anchors",
               "test/test_synthetic_appearance_presets.py::test_ga10_routine_placement_and_its_stop",
               "test/test_synthetic_appearance_presets.py::test_ga11_routine_territories",
               "test/test_synthetic_appearance_presets.py::test_ga18_routine_projection_truth",
               "test/test_synthetic_appearance_coverage.py::"
               "test_ga13_kappa_and_the_rim_background_on_the_routine_presets",
               "test/test_synthetic_appearance_coverage.py::test_ga17_types_and_molecules_on_the_routine_presets"],
     "reason": ("Each routine appearance preset is generated, banked and placed once per process; its first user "
                "pays 5.61 to 9.82 s and the others 0.02 to 1.78 s, so all are slow by rule 5."),
     "evidence": ("runs/W-530/20261020T132102Z-6834ae5f/results/measurements/app-routine-users.log (one CPU, "
                  "--durations=0): '9.82s call "
                  "test/test_synthetic_appearance_presets.py::test_ga8_routine_labels_equal_the_oracle_on_every_grid"
                  "[appearance-culture-v1]'")},
)

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


def exemption_of(node: str, exemptions=EXEMPTIONS) -> str | None:
    """The name of the entry that covers ``node`` (any parametrization of a listed test), or None."""
    test = node.split("[", 1)[0]
    return next((entry["name"] for entry in exemptions if test in entry["tests"]), None)


def drift(durations: dict[str, dict[str, float]], outcomes: dict[str, str], items: dict[str, dict],
          slow_at: float = SLOW_AT, fast_under: float = FAST_UNDER, exemptions=EXEMPTIONS) -> dict:
    """The drift report; ``items`` maps a node ID to {"markers": [...]}."""
    ran = {n for n in set(durations) | set(outcomes) if outcomes.get(n) != "SKIPPED"}
    report = {"slow_at": slow_at, "fast_under": fast_under, "tests": len(ran),
              "exemptions": [{k: e[k] for k in ("name", "source", "reason", "evidence")} for e in exemptions],
              "unmarked_at_or_over": [], "slow_under": [], "slow_under_exempt": [], "not_collected": []}
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
            entry = exemption_of(node, exemptions)
            if entry:
                report["slow_under_exempt"].append(dict(row, exemption=entry))
            else:
                report["slow_under"].append(row)
    for name in ("unmarked_at_or_over", "slow_under", "slow_under_exempt"):
        report[name].sort(key=lambda r: (-r["seconds"], r["nodeid"]))
    return report


def text(report: dict) -> str:
    """The report as plain text."""
    lines = [f"{report['tests']} tests ran (skipped tests left out)", "",
             f"Without slow, {report['slow_at']:g} s or more: {len(report['unmarked_at_or_over'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}  {r['phases']}" for r in report["unmarked_at_or_over"]]
    lines += ["", f"slow, under {report['fast_under']:g} s: {len(report['slow_under'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}" for r in report["slow_under"]]
    lines += ["", f"Left out (slow, under {report['fast_under']:g} s, covered by an exemption): "
              f"{len(report['slow_under_exempt'])}"]
    lines += [f"  {r['seconds']:7.2f} s  {r['nodeid']}  exemption {r['exemption']}"
              for r in report["slow_under_exempt"]]
    lines += ["", f"Exemptions: {len(report['exemptions'])}"]
    lines += [f"  {e['name']}: {e['source']}; {e['reason']} Evidence: {e['evidence']}" for e in report["exemptions"]]
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
