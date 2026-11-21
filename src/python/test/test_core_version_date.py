"""One version date per process tree (W-914): a modified checkout's version ends in ``.d<YYYYMMDD>``, which
``_records._checkout_version`` takes from ``SOURCE_DATE_EPOCH``, else from the clock in UTC.

``_records.fix_version_date()`` sets the variable at the top of a process tree when it is unset, and every child
inherits it, so the processes of one test session or workflow run record one version, also when the run crosses
00:00 UTC. The two-date cases run a parent and a child over a copy of ``_records.py`` in a throwaway modified checkout,
each with its clock set to one side of a UTC midnight. The session case runs a pytest session with ``-n 2`` over a copy
of this suite's ``conftest.py``.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from .test_core_code_identity import EPOCH, git, needs_git, repository

pytestmark = [pytest.mark.core, pytest.mark.contract]

#: 2026-11-20 23:59:30 UTC and, a minute later, 2026-11-21 00:00:30 UTC: 19:00 EST falls between them.
BEFORE, AFTER = 1795219170, 1795219230
CONFTEST = Path(__file__).resolve().with_name("conftest.py")
SOURCE = Path(__file__).resolve().parents[1]

# The parent: its clock before midnight, then (with `fix`) the hook, its record, and a child whose clock is after
# midnight and which writes its own record. Each record is the `software` section that every record of the package
# carries.
PARENT = """
import json, os, subprocess, sys, time
time.time = lambda: {before}
from starfinder import _records
if sys.argv[1] == "fix":
    _records.fix_version_date()
_records.write_json({{"software": _records.software()}}, "parent.json")
child = ("import time; time.time = lambda: {after}; from starfinder import _records; "
         "_records.write_json({{'software': _records.software()}}, 'child.json')")
subprocess.run([sys.executable, "-c", child], check=True)
print(json.dumps(os.environ.get("SOURCE_DATE_EPOCH")))
"""


def without_date(**extra):
    """This process's environment without ``SOURCE_DATE_EPOCH`` (the session sets it), with ``extra`` added."""
    env = {k: v for k, v in os.environ.items() if k != "SOURCE_DATE_EPOCH" and not k.startswith("PYTEST_")}
    return dict(env, **extra)


def parent_and_child(repo, mode, **extra):
    """(the parent's SOURCE_DATE_EPOCH at its end, the parent's record, the child's record) of one modified tree."""
    src = repo / "src" / "python"
    env = without_date(PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1", **extra)
    done = subprocess.run([sys.executable, "-c", PARENT.format(before=BEFORE, after=AFTER), mode], cwd=repo,
                          env=env, capture_output=True, text=True, check=True)
    records = [json.loads((repo / name).read_text()) for name in ("parent.json", "child.json")]
    texts = [(repo / name).read_bytes() for name in ("parent.json", "child.json")]
    return json.loads(done.stdout), records, texts


@pytest.fixture
def modified(tmp_path):
    repo = repository(tmp_path, {"v0.3.1": 1}, 3)
    (repo / "notes.txt").write_text("changed")
    return repo, git(repo, "rev-parse", "--short", "HEAD")[:9]


@needs_git
def test_a_parent_and_its_child_on_two_utc_dates_record_one_version(modified):
    repo, short = modified
    value, (parent, child), texts = parent_and_child(repo, "fix")
    assert value == str(BEFORE)
    assert parent["software"]["code"]["version"] == f"0.3.2.dev2+g{short}.d20261120"
    assert parent == child
    assert texts[0] == texts[1]


@needs_git
def test_without_the_variable_each_process_takes_the_date_of_its_clock(modified):
    # Today's behavior, unchanged: nothing sets the variable, so each process reads its own clock.
    repo, short = modified
    value, (parent, child), texts = parent_and_child(repo, "clock")
    assert value is None
    assert parent["software"]["code"]["version"] == f"0.3.2.dev2+g{short}.d20261120"
    assert child["software"]["code"]["version"] == f"0.3.2.dev2+g{short}.d20261121"
    assert texts[0] != texts[1]


@needs_git
def test_the_hook_keeps_a_value_the_caller_set(modified):
    repo, short = modified
    value, (parent, child), _ = parent_and_child(repo, "fix", SOURCE_DATE_EPOCH=EPOCH)
    assert value == EPOCH
    assert {parent["software"]["code"]["version"], child["software"]["code"]["version"]} == {
        f"0.3.2.dev2+g{short}.d20261110"}


def test_fix_version_date_sets_the_variable_only_when_it_is_unset(monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    from starfinder import _records
    before = int(time.time())
    value = _records.fix_version_date()
    assert before <= int(value) <= int(time.time())
    assert os.environ["SOURCE_DATE_EPOCH"] == value
    monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)
    assert _records.fix_version_date() == EPOCH == os.environ["SOURCE_DATE_EPOCH"]


# The inner session: the controller's value at session start, before pytest-xdist starts its workers (its own
# pytest_sessionstart is trylast), and, from each test, the worker's value and that of a subprocess it starts.
PROBE_PLUGIN = """
import json, os
from pathlib import Path
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    if not hasattr(session.config, "workerinput"):
        Path(os.environ["W914_OUT"], "controller.json").write_text(json.dumps(os.environ.get("SOURCE_DATE_EPOCH")))
"""
PROBE_TESTS = """
import json, os, subprocess, sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.core

@pytest.mark.parametrize("number", range(4))
def test_probe(number):
    child = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('SOURCE_DATE_EPOCH'))"],
                           capture_output=True, text=True, check=True).stdout.strip()
    seen = {"worker": os.environ.get("PYTEST_XDIST_WORKER"), "value": os.environ.get("SOURCE_DATE_EPOCH"),
            "child": child}
    Path(os.environ["W914_OUT"], f"{number}.json").write_text(json.dumps(seen))
"""


def session(folder, **extra):
    """Start a pytest session with ``-n 2`` over this suite's conftest.py and four probe tests; return the process."""
    folder.mkdir()
    out = folder / "out"
    out.mkdir()
    shutil.copy(CONFTEST, folder / "conftest.py")
    (folder / "w914_probe.py").write_text(PROBE_PLUGIN)
    (folder / "test_probe.py").write_text(PROBE_TESTS)
    (folder / "pytest.ini").write_text("[pytest]\nmarkers =\n    core: the probe's subsystem\n")
    env = without_date(PYTHONPATH=os.pathsep.join([str(SOURCE), str(folder)]), PYTHONDONTWRITEBYTECODE="1",
                       W914_OUT=str(out), **extra)
    command = [sys.executable, "-m", "pytest", "-p", "w914_probe", "-p", "no:cacheprovider", "-n", "2", "-q",
               "test_probe.py"]
    return subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def seen(folder):
    """(the controller's value, [each probe's record])."""
    out = folder / "out"
    return (json.loads((out / "controller.json").read_text()),
            [json.loads((out / f"{n}.json").read_text()) for n in range(4)])


@pytest.mark.slow
def test_a_session_with_workers_and_its_subprocesses_see_one_value(tmp_path):
    pytest.importorskip("xdist", reason="pytest-xdist is not installed")
    assert os.environ.get("SOURCE_DATE_EPOCH")  # this session's own, from conftest.py
    started = int(time.time())
    unset = session(tmp_path / "unset")
    caller = session(tmp_path / "caller", SOURCE_DATE_EPOCH=EPOCH)
    outputs = {name: process.communicate(timeout=300)[0] for name, process in (("unset", unset), ("caller", caller))}
    ended = int(time.time())
    for name, process in (("unset", unset), ("caller", caller)):
        assert process.returncode == 0, outputs[name]
    # Unset by the caller: the session's conftest.py set it before the workers started, and all inherited it.
    controller, probes = seen(tmp_path / "unset")
    assert controller is not None and started <= int(controller) <= ended
    assert {p["worker"] for p in probes} == {"gw0", "gw1"}
    assert {(p["value"], p["child"]) for p in probes} == {(controller, controller)}
    # Set by the caller: kept everywhere.
    controller, probes = seen(tmp_path / "caller")
    assert controller == EPOCH
    assert {p["worker"] for p in probes} == {"gw0", "gw1"}
    assert {(p["value"], p["child"]) for p in probes} == {(EPOCH, EPOCH)}
