"""Snakemake subprocesses for the workflow's engine tests: real runs of ``workflow/Snakefile`` with one configuration.

Snakemake 9 comes from the ``workflow`` dependency group, beside the interpreter; ``_snakemake`` skips with the reason
"Snakemake not installed" where it is absent. Each Snakemake process runs with one numerical thread and the
``starfinder`` command of this environment first on PATH. ``TRIGGERS`` are the rerun triggers of the engine record's
profile setting, and ``downstream`` the tasks of a manifest that need given tasks, transitively.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from starfinder._execution import THREAD_VARIABLES

from .workflow_fixture import ROOT

SNAKEFILE = ROOT / "workflow" / "Snakefile"
TRIGGERS = ["--rerun-triggers", "mtime", "params", "input"]


def _snakemake():
    executable = Path(sys.executable).with_name("snakemake")
    if importlib.util.find_spec("snakemake") is None or not executable.is_file():
        pytest.skip("Snakemake not installed")
    return executable


def snakemake(work, config, *arguments):
    """Run Snakemake on ``workflow/Snakefile`` with one configuration file; return the completed process."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **dict.fromkeys(THREAD_VARIABLES, "1"),
                       ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")
    environment["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")])
    command = [str(_snakemake()), "-s", str(SNAKEFILE), "--directory", str(Path(work) / "snakemake"), "--cores", "1",
               "--nocolor", "--printshellcmds", "--configfile", str(config), *arguments]
    return subprocess.run(command, capture_output=True, text=True, env=environment)


def downstream(plan, tasks):
    """``tasks`` and every task that needs one of them, transitively (the manifest's order respects needs)."""
    found = set(tasks)
    for task in plan["tasks"]:
        if found.intersection(task["needs"]):
            found.add(task["id"])
    return found
