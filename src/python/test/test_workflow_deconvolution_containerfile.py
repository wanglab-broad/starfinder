"""IB1 of the deconvolution validation design (W-936): the ``Containerfile`` pins a reproducible deconwolf build.

Two builds of the ``build-deconwolf`` stage on different days gave different binaries, because GCC wrote the build's
date into ``dw``, and the wisdom key holds the binary's SHA-256. The stage now sets ``SOURCE_DATE_EPOCH``, takes its
apt packages from one ``snapshot.debian.org`` timestamp, and the ``runtime`` stage checks the binary against a pinned
``DECONWOLF_SHA256``. These tests read the file as text and build no image; IM5 builds the stage on two days.
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

CONTAINERFILE = Path(__file__).resolve().parents[3] / "Containerfile"
#: The deconwolf v0.4.5 commit's time (ca062c49, 2025-03-12T08:25:06Z), the mtime of every entry of the tag's archive.
EPOCH = "1741767906"
SNAPSHOT = re.compile(r"snapshot\.debian\.org/archive/(debian|debian-security)/(\d{8}T\d{6}Z)")


def stage(name):
    """The lines of one Containerfile stage, backslash continuations joined and comments left out."""
    text = CONTAINERFILE.read_text().replace("\\\n", " ")
    sections = re.split(r"^(?=FROM )", text, flags=re.MULTILINE)
    section = next(s for s in sections if re.match(rf"FROM .* AS {re.escape(name)}$", s.splitlines()[0]))
    return [line for line in section.splitlines()[1:] if line.strip() and not line.startswith("#")]


def index(lines, prefix, contains=""):
    """The position of the one line that starts with ``prefix`` and contains ``contains``."""
    found = [i for i, line in enumerate(lines) if line.startswith(prefix) and contains in line]
    assert len(found) == 1, (prefix, contains, found)
    return found[0]


@pytest.mark.skipif(not CONTAINERFILE.is_file(), reason="the Containerfile is not in this tree")
def test_ib1_the_deconwolf_stage_is_a_reproducible_build():
    lines = stage("build-deconwolf")
    build = index(lines, "RUN cmake ")
    # The build's date reaches the binary through GCC's __DATE__, so the epoch is set for the CMake step.
    assert index(lines, "ARG SOURCE_DATE_EPOCH") < build
    assert lines[index(lines, "ARG SOURCE_DATE_EPOCH")] == f"ARG SOURCE_DATE_EPOCH={EPOCH}"
    assert index(lines, "ENV SOURCE_DATE_EPOCH") < build
    assert lines[index(lines, "ENV SOURCE_DATE_EPOCH")] == "ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
    # The apt sources are the snapshot's, set before the stage's own apt-get: the debian URI (bookworm and
    # bookworm-updates) and the debian-security URI (bookworm-security) at one timestamp.
    sources = index(lines, "RUN ", "/etc/apt/sources.list.d/debian.sources")
    assert sources < index(lines, "RUN apt-get update")
    found = SNAPSHOT.findall(lines[sources])
    assert sorted(kind for kind, _ in found) == ["debian", "debian-security"]
    assert len({timestamp for _, timestamp in found}) == 1
    assert "Check-Valid-Until: no" in lines[sources]
    assert "deb.debian.org" in lines[sources]


@pytest.mark.skipif(not CONTAINERFILE.is_file(), reason="the Containerfile is not in this tree")
def test_ib1_only_the_deconwolf_stage_uses_the_snapshot_and_the_runtime_stage_checks_the_pinned_binary():
    text = CONTAINERFILE.read_text()
    stages = re.findall(r"^FROM .* AS (\S+)$", text, flags=re.MULTILINE)
    assert "build-deconwolf" in stages
    for name in stages:
        uses = any("snapshot.debian.org" in line for line in stage(name))
        assert uses == (name == "build-deconwolf"), name
        # The epoch is the deconwolf stage's alone.
        assert any("SOURCE_DATE_EPOCH" in line for line in stage(name)) == (name == "build-deconwolf"), name
    # The default of DECONWOLF_SHA256 is a SHA-256, and the runtime stage checks the binary against it.
    runtime = stage("runtime")
    pinned = runtime[index(runtime, "ARG DECONWOLF_SHA256")]
    assert re.fullmatch(r"ARG DECONWOLF_SHA256=[0-9a-f]{64}", pinned), pinned
    assert index(runtime, "ARG DECONWOLF_SHA256") < index(runtime, "RUN echo \"${DECONWOLF_SHA256}", "sha256sum -c -")
    assert any(line.startswith("RUN echo \"${DECONWOLF_SHA256}  /opt/deconwolf/bin/dw\" | sha256sum -c -")
               for line in runtime)
