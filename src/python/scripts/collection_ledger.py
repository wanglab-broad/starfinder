#!/usr/bin/env python3
"""Check that every test that left the collection between two revisions has a row in a ledger (W-737, W-750).

An issue that removes or moves tests saves ``pytest --collect-only -q -o addopts=`` from its base revision and from
its tip; a test id the base collects and the tip does not has left. The ledger is a tab-separated file with the
header ``test_id``, ``class``, ``reference``: one row per test that left, its class one of W-750's ``I``, ``M``,
``R`` and ``D``, and its reference the destination test, the test that keeps its assertion, or the decision. The
report states:

* the tests that left without a row (a failure: add the row or keep the test);
* the rows whose class is not one of the four or whose reference is empty;
* the rows for a test the tip still collects, or the base never collected;
* the rows whose reference names a test (``<file>::<name>``) that the tip does not collect (a parametrized test
  counts when the tip collects one of its cases).

Run from src/python::

    uv run python scripts/collection_ledger.py BASE.txt TIP.txt LEDGER.tsv [--output result.json]

The exit status is 0 when the ledger is complete and every row is valid, and 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CLASSES = ("I", "M", "R", "D")
COLUMNS = ["test_id", "class", "reference"]


def collected(path):
    """The test ids of a ``pytest --collect-only -q`` listing (its lines that hold ``::``), in order."""
    return [line.strip() for line in Path(path).read_text().splitlines() if "::" in line and not line.startswith(" ")]


def read_ledger(path):
    """The ledger's rows as dicts with COLUMNS."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path}: the header must be {' '.join(COLUMNS)}, not {reader.fieldnames}")
        return [{k: (row[k] or "").strip() for k in COLUMNS} for row in reader]


def _collects(ids, test):
    """Whether ``ids`` hold ``test`` or a parametrized case of it."""
    return test in ids or any(i.startswith(test + "[") for i in ids)


def check(base, tip, rows):
    """The report of a ledger: ``base`` and ``tip`` are lists of test ids, ``rows`` the ledger's rows."""
    base_ids, tip_ids = set(base), set(tip)
    left = [t for t in base if t not in tip_ids]
    ledgered = {row["test_id"] for row in rows}
    report = {
        "base": len(base), "tip": len(tip), "left": left, "added": [t for t in tip if t not in base_ids],
        "left_without_row": [t for t in left if t not in ledgered],
        "invalid_rows": [row for row in rows if row["class"] not in CLASSES or not row["reference"]],
        "rows_still_collected": [row["test_id"] for row in rows if row["test_id"] in tip_ids],
        "rows_never_collected": [row["test_id"] for row in rows if row["test_id"] not in base_ids],
        "references_not_collected": [row for row in rows if "::" in row["reference"]
                                     and not _collects(tip_ids, row["reference"])],
    }
    report["ok"] = not any(report[k] for k in ("left_without_row", "invalid_rows", "rows_still_collected",
                                               "rows_never_collected", "references_not_collected"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="pytest --collect-only -q output at the issue's base")
    parser.add_argument("tip", type=Path, help="the same at the issue's tip")
    parser.add_argument("ledger", type=Path, help="the ledger (tab-separated: test_id, class, reference)")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)
    report = check(collected(args.base), collected(args.tip), read_ledger(args.ledger))
    text = json.dumps(report, indent=1) + "\n"
    if args.output:
        args.output.write_text(text)
    sys.stdout.write(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
