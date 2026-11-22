"""One version date per workflow run (W-914): every process of a run gets the ``SOURCE_DATE_EPOCH`` of the engine's
head.

A modified checkout's version takes its ``.d<YYYYMMDD>`` part from that variable, else from the clock, so a run whose
processes read their own clocks across 00:00 UTC records two versions. Snakemake's main process sets it when it is unset
(``rules/tasks.smk``) and its local jobs inherit it; Nextflow's ``env`` scope gives the head's value to every task
(``nextflow/nextflow.config``). Each run here starts every process through a stand-in that logs the value it sees: the
``starfinder`` command first on PATH (the plan, Nextflow's verify, Snakemake's task commands) and Nextflow's stub
runner. The Snakemake jobs run Nextflow's stub runner in place of the task, so no task computes. A resumed Nextflow run
with another value finds every task in its cache: the ``env`` scope is not part of a task's hash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import workflow_fixture as F
from . import workflow_nextflow as N
from . import workflow_variants as V
from .test_core_code_identity import EPOCH
from .workflow_snakemake import SNAKEFILE, _snakemake

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.slow]

STUB = F.ROOT / "nextflow" / "bin" / "stub_task.sh"
# The stand-in `starfinder`: logs the value and what it runs, then the real command for the plan and verify, or the stub
# runner for a task command (manifest, task id, token, attempt).
STARFINDER = """#!{python}
import json, os, sys
arguments = sys.argv[1:]
planning = arguments[:2] in (["workflow", "plan"], ["workflow", "verify"])
value = lambda name: arguments[arguments.index(name) + 1]
what = arguments[:2] if planning else ["stub", value("--task")]
with open(os.environ["W914_LOG"], "a") as handle:
    handle.write(json.dumps([os.environ.get("SOURCE_DATE_EPOCH"), what]) + "\\n")
if planning:
    os.execv(sys.executable, [sys.executable, "-m", "starfinder", *arguments])
os.execv("{stub}", ["{stub}", value("--manifest"), value("--task"), "none", "1"])
"""
# Nextflow's stub runner: logs the value, then the real one.
RUNNER = """#!{python}
import json, os, sys
with open(os.environ["W914_LOG"], "a") as handle:
    handle.write(json.dumps([os.environ.get("SOURCE_DATE_EPOCH"), ["stub", sys.argv[2]]]) + "\\n")
os.execv("{stub}", ["{stub}", *sys.argv[1:]])
"""


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    work = tmp_path_factory.mktemp("version-date")
    F.golden().write_fixture(work)
    F.write_supplied(work)
    return work


def stand_ins(folder):
    """(the folder of the stand-in ``starfinder``, the stub runner, the log)."""
    tools = folder / "tools"
    tools.mkdir(parents=True)
    for path, text in ((tools / "starfinder", STARFINDER), (folder / "runner.py", RUNNER)):
        path.write_text(text.format(python=sys.executable, stub=STUB))
        path.chmod(0o755)
    return tools, folder / "runner.py", folder / "processes.jsonl"


def configuration(work, name):
    config = N.small_resources(V.variants(work)["reference-only"])
    config.update(output_id=name, selected_fovs=["FOV_001"])
    return F.write_config(work, name, config)


def values(log):
    """[(value, what ran)] of every process, in order."""
    return [tuple(json.loads(line)) for line in log.read_text().splitlines()]


def check(seen, manifest, caller, started, ended):
    tasks = [what[1] for _, what in seen if what[0] == "stub"]
    assert sorted(tasks) == sorted(task["id"] for task in manifest["tasks"])
    assert ["workflow", "plan"] in [what for _, what in seen]
    (value,) = {value for value, _ in seen}
    if caller:
        assert value == EPOCH
    else:
        assert value is not None and started <= int(value) <= ended


def date(monkeypatch, caller):
    if caller:
        monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)
    else:
        monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)


@pytest.mark.parametrize("caller", [False, True], ids=["unset", "caller"])
def test_every_process_of_a_snakemake_run_sees_one_value(tiny, tmp_path, monkeypatch, caller):
    _snakemake()
    date(monkeypatch, caller)
    name = f"snakemake-{'caller' if caller else 'unset'}"
    config = configuration(tiny, name)
    tools, _, log = stand_ins(tmp_path)
    env = dict(os.environ, W914_LOG=str(log), PYTHONDONTWRITEBYTECODE="1",
               PATH=os.pathsep.join([str(tools), str(Path(sys.executable).parent), os.environ.get("PATH", "")]))
    command = [str(_snakemake()), "-s", str(SNAKEFILE), "--directory", str(tmp_path / "snakemake"), "--cores", "1",
               "--nocolor", "--configfile", str(config)]
    started = int(time.time())
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    ended = int(time.time())
    assert done.returncode == 0, done.stdout[-5000:] + done.stderr[-5000:]
    manifest = json.loads((F.output_root(tiny, name) / "workflow" / "manifest.json").read_text())
    check(values(log), manifest, caller, started, ended)


@pytest.mark.nextflow
@pytest.mark.parametrize("caller", [False, True], ids=["unset", "caller"])
def test_every_process_of_a_nextflow_run_sees_one_value(tiny, tmp_path, monkeypatch, caller):
    N.executable()
    date(monkeypatch, caller)
    name = f"nextflow-{'caller' if caller else 'unset'}"
    config = configuration(tiny, name)
    tools, runner, log = stand_ins(tmp_path)
    path = os.pathsep.join([str(tools), str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    started = int(time.time())
    done, _ = N.run(tmp_path, config, "--stub_runner", str(runner), environment={"W914_LOG": str(log), "PATH": path})
    ended = int(time.time())
    assert done.returncode == 0, done.stdout + done.stderr
    seen = values(log)
    assert [what for _, what in seen].count(["workflow", "verify"]) == 2
    check(seen, N.manifest(config), caller, started, ended)
    if not caller:
        # The env scope is not part of a task's hash: a resumed run with another value runs no task again.
        monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)
        done, rows = N.run(tmp_path, config, "--stub_runner", str(runner), "-resume",
                           environment={"W914_LOG": str(log), "PATH": path})
        assert done.returncode == 0, done.stdout + done.stderr
        assert {r["status"] for r in N.tasks(rows)} == {"CACHED"}
        assert len(N.tasks(rows)) == len(N.manifest(config)["tasks"])
