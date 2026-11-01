"""Nextflow subprocesses and retained traces for the workflow's engine tests."""
from __future__ import annotations

import csv
import hashlib
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from . import workflow_examples as X


def executable():
    path = shutil.which("nextflow")
    if path is None:
        pytest.skip("Nextflow not installed")
    return path


def run(work, config, *arguments, stub=True, environment=None):
    """One invocation in a persistent local cache; capture logs and a unique trace even on refusal."""
    executable_path = executable()
    folder = Path(work) / "nextflow"
    folder.mkdir(parents=True, exist_ok=True)
    index = len(list(folder.glob("run-*.log")))
    prefix = folder / f"run-{index:02d}"
    trace = prefix.with_suffix(".tsv")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.update(environment or {})
    command = [executable_path, "-log", str(prefix.with_suffix(".nextflow.log")), "run",
               str(X.ROOT / "nextflow" / "main.nf"), "-ansi-log", "false", "-profile", "local",
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
