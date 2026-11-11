"""W-831: the Nextflow tests skip, naming the reason, without Nextflow 26.10.4 or newer on Java 17 or newer.

Known answers of the helper that reads ``nextflow info``, on stubbed text: no process starts and nothing touches the network.
"""
from __future__ import annotations

import pytest

from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract]


def info(version, java):
    return (f"  Version: {version} build 14435\n  Created: 26-10-2026 21:11 UTC (17:11 EDT)\n  System: Linux 5.15\n"
            f"  Runtime: Groovy 4.0.33 on OpenJDK 64-Bit Server VM {java}\n  Encoding: UTF-8 (UTF-8)\n")


@pytest.mark.parametrize("report, expected", [
    (info("26.10.4", "23.0.2-internal-adhoc.conda.src"), ((26, 10, 4), 23)),
    (info("25.04.7", "17.0.9+9-LTS"), ((25, 4, 7), 17)),
    (info("26.10.4", "1.8.0_392-b08"), ((26, 10, 4), 8)),
    (info("26.10.4", "11.0.19+7"), ((26, 10, 4), 11)),
    ("", (None, None)),
    ("nothing useful here", (None, None)),
])
def test_versions_are_read_from_the_report_of_nextflow_info(report, expected):
    assert N.versions(report) == expected


@pytest.mark.parametrize("path, report, reason", [
    ("/bin/nextflow", info("26.10.4", "23.0.2"), None),
    ("/bin/nextflow", info("26.11.0", "17.0.1"), None),
    ("/bin/nextflow", info("27.1.0", "21"), None),
    (None, "", "Nextflow not installed"),
    ("/bin/nextflow", info("26.10.3", "23.0.2"), "Nextflow 26.10.3 is older than 26.10.4"),
    ("/bin/nextflow", info("25.10.4", "23.0.2"), "Nextflow 25.10.4 is older than 26.10.4"),
    ("/bin/nextflow", info("26.10.4", "11.0.19"), "Nextflow runs on Java 11, older than 17"),
    ("/bin/nextflow", info("26.10.3", "11.0.19"), "Nextflow 26.10.3 is older than 26.10.4"),
    ("/bin/nextflow", "", "the version of Nextflow or of its Java could not be read from `nextflow info`"),
])
def test_the_engine_tests_run_only_with_the_minimum_versions(path, report, reason):
    assert N.unavailable(path, report) == reason


def test_a_java_older_than_the_minimum_skips_a_test_that_asks_for_the_executable(monkeypatch):
    """``executable()`` skips, with the reason, on an old Java; the stub names a path and a report, so nothing starts."""
    monkeypatch.setattr(N.shutil, "which", lambda name: "/bin/nextflow")
    monkeypatch.setattr(N, "_report", lambda path: info("26.10.4", "11.0.19"))
    with pytest.raises(pytest.skip.Exception, match="Java 11, older than 17"):
        N.executable()
