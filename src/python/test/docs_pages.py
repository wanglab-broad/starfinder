"""The parts of a documentation page or design record that a docs-contract test parses, found by their labels (W-602).

A table or passage that a test parses carries a MyST target label right before it, ``(label)=``
(docs/contributing.md, "Test markers"). The tests find it by that label and never by heading text, so a
page's headings may change freely; the label and the row shape of its table stay. This module names no
page: the docs-contract modules pass the page's text. Pages of the site are under ``docs/``; design
records, which the site does not build, are under ``design/`` (design/README.md).
"""
from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[3] / "docs"
DESIGN = Path(__file__).resolve().parents[3] / "design"
HEADING = re.compile(r"(#{1,6}) ")
FENCE = re.compile(r"(`{3,}|~{3,})")


def read(page: str) -> str:
    """The text of a page, by its path under docs/."""
    return (DOCS / page).read_text(encoding="utf-8")


def design(record: str) -> str:
    """The text of a design record, by its path under design/."""
    return (DESIGN / record).read_text(encoding="utf-8")


def _label(lines: list[str], label: str) -> int:
    """The index of the one line ``(label)=``."""
    found = [i for i, line in enumerate(lines) if line.strip() == f"({label})="]
    assert len(found) == 1, f"({label})= occurs {len(found)} times"
    return found[0]


def _target(lines: list[str], label: str) -> int:
    """The index of the first non-blank line after the label."""
    start = _label(lines, label) + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    return start


def table(text: str, label: str) -> str:
    """The rows of the table right after the label (blank lines between them only), one per line."""
    lines = text.splitlines()
    start = end = _target(lines, label)
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    assert end > start, f"no table follows ({label})="
    return "\n".join(lines[start:end]) + "\n"


def section(text: str, label: str) -> str:
    """The section whose heading follows the label: its heading line up to the next heading of its level or higher,
    outside code fences."""
    lines = text.splitlines()
    start = _target(lines, label)
    heading = HEADING.match(lines[start]) if start < len(lines) else None
    assert heading, f"no heading follows ({label})="
    level, fence, end = len(heading.group(1)), None, start + 1
    while end < len(lines):
        line = lines[end]
        mark = FENCE.match(line)
        if fence is None and mark:
            fence = mark.group(1)[0]
        elif fence is not None and line.startswith(fence * 3):
            fence = None
        elif fence is None and (match := HEADING.match(line)) and len(match.group(1)) <= level:
            break
        end += 1
    return "\n".join(lines[start:end]) + "\n"


def between(text: str, start: str, end: str) -> str:
    """The passage from the label ``start`` to the label ``end``."""
    lines = text.splitlines()
    first, last = _target(lines, start), _label(lines, end)
    assert first < last, f"({start})= is not before ({end})="
    return "\n".join(lines[first:last]) + "\n"
