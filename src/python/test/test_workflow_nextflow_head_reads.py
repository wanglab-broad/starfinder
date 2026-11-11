"""The Nextflow head reads work-directory outputs through the Path API, never through java.io (W-834).

On Google Batch the work directory is ``gs://``: a process output is a ``CloudStoragePath``, and ``toFile()`` on it
throws "GCS objects aren't available locally" (W-832's route-B spike, at the head's first read of ``plan.json``).
A static scan is enough: the sites are few, and the engine tests cover their behaviour locally.
"""
from __future__ import annotations

import re

import pytest

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

SOURCES = ["nextflow/main.nf", *(f"nextflow/modules/{p.name}" for p in sorted((X.ROOT / "nextflow/modules").glob("*.nf"))),
           *(f"nextflow/lib/{p.name}" for p in sorted((X.ROOT / "nextflow/lib").glob("*.groovy")))]

# The only java.io.File sites. Each reads the manifest, whose path is under roots.output, which every host mounts, so a
# local file is right there; none reads a process output of the work directory. The count is per file.
ALLOWED_NEW_FILE = {"nextflow/main.nf": 2, "nextflow/lib/TaskCommands.groovy": 2}
NEW_FILE = re.compile(r"\bnew\s+(?:java\.io\.)?File\s*\(")
TO_FILE = re.compile(r"\.toFile\b|\bjava\.io\.")


def code_lines(relative):
    """Each line of a source with its comments blanked, with the line number and the comment text before it."""
    lines = (X.ROOT / relative).read_text().splitlines()
    for number, line in enumerate(lines, 1):
        yield number, re.sub(r"(^|\s)//.*", "", line), lines[max(0, number - 4):number]


def test_no_work_directory_output_goes_through_java_io():
    found = [f"{relative}:{number}: {line.strip()}" for relative in SOURCES
             for number, line, _ in code_lines(relative) if TO_FILE.search(line)]
    assert not found, "Read a process output with Path (p.text, p.readLines(), p.withInputStream), not toFile():\n" + "\n".join(found)


def test_new_file_only_where_the_manifest_is_read_from_the_output_root():
    sites = {}
    for relative in SOURCES:
        for number, line, before in code_lines(relative):
            if NEW_FILE.search(line):
                sites.setdefault(relative, []).append(number)
                assert any("java.io.File" in text and "roots.output" in text for text in before), (
                    f"{relative}:{number}: say in a comment why java.io.File is right here (a path under roots.output)")
    assert {k: len(v) for k, v in sites.items()} == ALLOWED_NEW_FILE
