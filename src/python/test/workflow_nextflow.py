"""Nextflow subprocesses and retained traces for the workflow's engine tests."""
from __future__ import annotations

import csv
import functools
import hashlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from starfinder._execution import THREAD_VARIABLES

from . import workflow_examples as X


#: The engine tests need Nextflow 26.10.4 or newer, which needs Java 17 or newer (``manifest.nextflowVersion`` of
#: ``nextflow/nextflow.config``).
NEXTFLOW_MINIMUM = (26, 10, 4)
JAVA_MINIMUM = 17


def versions(report):
    """(Nextflow version tuple, Java major version) from the text of ``nextflow info``; None for what it does not give.

    Java is the one Nextflow runs on (``Runtime: Groovy 4.0.33 on OpenJDK 64-Bit Server VM 23.0.2-...``), which is not
    always the ``java`` of ``PATH``: the install may bring its own.
    """
    found = re.search(r"Version:\s*(\d+(?:\.\d+)*)", report)
    runtime = re.search(r"Runtime:.*?\bVM\s+(\d+(?:\.\d+)*)", report)
    nextflow = tuple(int(n) for n in found.group(1).split(".")) if found else None
    java = None
    if runtime:
        parts = [int(n) for n in runtime.group(1).split(".")]
        java = parts[1] if parts[0] == 1 and len(parts) > 1 else parts[0]  # Java 8 reports 1.8.0
    return nextflow, java


def unavailable(path, report):
    """Why the engine tests cannot run, or None: no Nextflow, or a Nextflow or a Java older than the minimum."""
    if path is None:
        return "Nextflow not installed"
    nextflow, java = versions(report or "")
    if nextflow is None or java is None:
        return "the version of Nextflow or of its Java could not be read from `nextflow info`"
    if nextflow < NEXTFLOW_MINIMUM:
        return f"Nextflow {'.'.join(map(str, nextflow))} is older than {'.'.join(map(str, NEXTFLOW_MINIMUM))}"
    if java < JAVA_MINIMUM:
        return f"Nextflow runs on Java {java}, older than {JAVA_MINIMUM}"
    return None


@functools.cache
def _report(path):
    """The text of ``nextflow info``, taken once per session (a selection starts Nextflow once, not once per test)."""
    try:
        done = subprocess.run([path, "info"], capture_output=True, text=True, timeout=120,
                              env=dict(os.environ, NXF_OFFLINE="true", NXF_PLUGINS_DEFAULT="false"))
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout


def executable():
    path = shutil.which("nextflow")
    reason = unavailable(path, _report(path) if path else "")
    if reason:
        pytest.skip(reason)
    return path


def run(work, config, *arguments, stub=True, environment=None, profile="local"):
    """One invocation in a persistent local cache; capture logs and a unique trace even on refusal."""
    executable_path = executable()
    folder = Path(work) / "nextflow"
    folder.mkdir(parents=True, exist_ok=True)
    index = len(list(folder.glob("run-*.log")))
    prefix = folder / f"run-{index:02d}"
    trace = prefix.with_suffix(".tsv")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # The Snakemake helper of the engine tests sets one thread everywhere; tasks record these variables in their
    # execution entry, so the Nextflow tasks need the same values whatever the caller's shell sets (W-828).
    env.update(dict.fromkeys(THREAD_VARIABLES, "1"), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")
    env.update(environment or {})
    # A caller's PATH (stand-in tools first) replaces the one above, and the interpreter's folder, which has `starfinder`
    # when the suite runs from an environment that is not activated, goes after the caller's entries (W-831).
    interpreter = str(Path(sys.executable).parent)
    if interpreter not in env["PATH"].split(os.pathsep):
        env["PATH"] += os.pathsep + interpreter
    command = [executable_path, "-log", str(prefix.with_suffix(".nextflow.log")), "run",
               str(X.ROOT / "nextflow" / "main.nf"), "-ansi-log", "false", "-profile", profile,
               "-with-trace", str(trace), "--config", str(config), *(["-stub-run"] if stub else []), *arguments]
    import yaml
    config_data = yaml.safe_load(Path(config).read_text())
    input_root = Path(config_data["root_input_path"]) / config_data["dataset_id"] / config_data["sample_id"]
    # Inventory fixture bytes before PLAN or any real command reads them.
    sources = [Path(config), *sorted(p for p in input_root.rglob("*") if p.is_file())]
    def named_files(value):
        if isinstance(value, dict):
            return [p for v in value.values() for p in named_files(v)]
        if isinstance(value, list):
            return [p for v in value for p in named_files(v)]
        if isinstance(value, str) and value.startswith("/"):
            pattern = value.replace("{fov_id}", "*").replace("{sample}", "*")
            return [Path(p) for p in glob.glob(pattern) if Path(p).is_file()]
        return []
    engine_sources = [p for p in (X.ROOT / "nextflow").rglob("*")
                      if p.suffix in (".nf", ".config", ".groovy", ".sh")]
    sources = sorted(set([*sources, *named_files(config_data), *engine_sources]))
    inventory = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    prefix.with_suffix(".inputs.json").write_text(json.dumps(inventory, indent=1) + "\n")
    done = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=900)
    prefix.with_suffix(".log").write_text("Command: " + repr(command) + "\n" + done.stdout + done.stderr)
    rows = []
    if trace.exists():
        with trace.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    # Controller and focused runs can request durable copies without using an SMB work directory.
    evidence = env.get("STARFINDER_NEXTFLOW_EVIDENCE")
    if evidence:
        destination = Path(evidence) / Path(work).parent.name / Path(work).name / prefix.name
        destination.mkdir(parents=True, exist_ok=True)
        for path in folder.glob(prefix.name + ".*"):
            target = destination / path.name
            shutil.copyfile(path, target)
            writes = env.get("STEP_WRITES")
            if writes:
                with Path(writes).open("a") as handle:
                    handle.write(str(target) + "\n")
    return done, rows


def tasks(rows):
    return [r for r in rows if r["tag"] not in ("-", "", None)]


def manifest(config):
    import yaml
    c = yaml.safe_load(Path(config).read_text())
    root = Path(c["root_output_path"]) / c["dataset_id"] / c["output_id"]
    return json.loads((root / "workflow" / "manifest.json").read_text())


def small_resources(config):
    """Bounded test execution settings; the worked-example's scientific blocks are unchanged."""
    from starfinder.workflow import TASK_BLOCKS
    settings = {"threads": 1, "memory_mb": 512, "runtime_min": 5}
    config["execution"] = {"defaults": dict(settings), "tasks": {k: dict(settings) for k in TASK_BLOCKS}}
    return config
