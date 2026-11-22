"""The runner of the evidence scripts that hold pytest tests: pytest on the script's own tests, and a record of their
outcomes (W-898; W-899).

``workflow_engine_evidence.py`` and ``orchestration_golden_evidence.py`` (W-898), and ``spot_finding_evidence.py`` and
``synthetic_scene_evidence.py`` (W-899), hold tests moved out of the suite as pytest tests, with their markers, fixtures
and assertions as they were. Each script's ``main`` calls ``run`` here,
which runs pytest on that file with the suite's configuration (``src/python/pyproject.toml``: its markers and warning
filters; not its ``addopts``, whose ``-m 'not extended'`` would deselect the moved extended tests) from the checkout
root, so that a case's node ID is ``benchmarks/<script>.py::<test>[<parameter>]``. The scripts load this file by its
location, as the other scripts load ``revision.py``.

``--output`` (new or empty, outside the repository) receives ``results.json`` (the revision of the checkout and its
dirty state, the Python, pytest, Snakemake and Nextflow it ran with, the evidence variables ``VARIABLES`` as set, and
per case its former node ID in the suite, its group of W-893's proposal, its outcome, its time and the message of a
failure or skip), pytest's ``junit.xml`` (``xunit1``, which keeps the values a case records with ``record_property``),
and the evidence the engine helpers keep (``STARFINDER_WORKFLOW_EVIDENCE`` and ``STARFINDER_NEXTFLOW_EVIDENCE``,
unless the caller sets them). pytest's temporary folders go where ``--basetemp`` names, else under the system's
temporary directory (``TMPDIR``), never in the output folder: the engines' work folders hold symlinks. A strict
expected failure that fails as expected is ``xfailed`` and counts as passed; one that passes is ``failed``, as pytest
reports it. Exit status 0 when every selected case passed (or failed as expected), 1 when one failed or raised, 2 when
none failed but one was skipped (a missing engine or extra, named in the message), 3 when nothing was selected.
``--list`` collects the cases without running them and prints each with its former node ID; it exits 1 if the
collection differs from the script's list of moved tests.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
CONFIGURATION = PACKAGE / "pyproject.toml"
#: The variables through which an evidence script finds what lies outside the checkout, recorded as set (W-899).
VARIABLES = ("STARFINDER_WEIGHTS_DIR", "STARFINDER_APPEARANCE_BANKS")


def parse(argv, description, epilog):
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--output", type=Path, help="new or empty directory outside the repository for results.json, "
                                                     "junit.xml and the engines' evidence")
    parser.add_argument("-k", dest="keyword", help="run only the cases whose node ID matches this pytest -k "
                                                    "expression; default: every case, in the order of the file")
    parser.add_argument("--case", action="append", default=[],
                        help="run only this case, the part of its node ID after '::' (repeatable; a parameter may hold "
                             "spaces)")
    parser.add_argument("--basetemp", type=Path, help="pytest's temporary folder (local disk; default: under TMPDIR)")
    parser.add_argument("--list", action="store_true", help="collect the cases and list them with their former node "
                                                            "IDs; runs no case")
    args = parser.parse_args(argv)
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is given")
    return args


def _revision():
    spec = importlib.util.spec_from_file_location("benchmark_revision", Path(__file__).resolve().parent / "revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revision()


def _command(*command, environment=None):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120,
                              env=dict(os.environ, **(environment or {})))
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}: {error}"
    return (done.stdout + done.stderr).strip()


def software():
    """The interpreter, pytest, and the engines the cases find: Snakemake beside the interpreter, Nextflow on PATH."""
    import pytest
    snakemake = Path(sys.executable).with_name("snakemake")
    nextflow = shutil.which("nextflow")
    return dict(python=sys.version.split()[0], executable=sys.executable, pytest=pytest.__version__,
                platform=platform.platform(), host=platform.node(),
                snakemake=_command(str(snakemake), "--version") if snakemake.is_file() else None,
                nextflow=nextflow, nextflow_info=_command(nextflow, "info", environment=dict(
                    NXF_OFFLINE="true", NXF_PLUGINS_DEFAULT="false")) if nextflow else None)


class Recorder:
    """A pytest plugin: the collected node IDs, and per case its outcome over setup, call and teardown."""

    def __init__(self):
        self.collected, self.cases = [], {}

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report):
        case = self.cases.setdefault(report.nodeid, dict(outcome="passed", seconds=0.0, message=""))
        case["seconds"] += report.duration
        if report.skipped and hasattr(report, "wasxfail"):
            case["outcome"] = "xfailed"
            case["message"] = report.wasxfail
        elif report.skipped and case["outcome"] == "passed":
            case["outcome"] = "skipped"
            case["message"] = str(report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr)
        elif report.failed:
            case["outcome"] = "failed" if report.when == "call" else "error"
            case["message"] = report.longreprtext[-6000:]


def _arguments(script, args, output=None):
    targets = [f"{script}::{case}" for case in args.case] or [str(script)]
    arguments = [*targets, "-c", str(CONFIGURATION), "--rootdir", str(ROOT), "-o", "addopts=",
                 "-p", "no:cacheprovider", "--strict-markers", "-v", "--tb=short", "--durations=0"]
    if args.keyword:
        arguments += ["-k", args.keyword]
    if args.basetemp:
        arguments += ["--basetemp", str(args.basetemp)]
    if output is not None:
        # xunit1 keeps the values a case records with record_property (the spot-finding checks' gated values, W-899).
        arguments += ["--junitxml", str(output / "junit.xml"), "-o", "junit_family=xunit1"]
    return arguments


def run(script, moved, origin_revision, argv, description, epilog):
    """Run (or with ``--list``, collect) the tests of ``script``. ``moved``: node ID in the script -> (the node ID it
    had in the suite at ``origin_revision``, its group of W-893's proposal)."""
    import pytest
    args = parse(argv, description, epilog)
    script = Path(script).resolve()
    if str(PACKAGE) not in sys.path[:1]:
        sys.path.insert(0, str(PACKAGE))
    recorder = Recorder()
    if args.list:
        with contextlib.redirect_stdout(sys.stderr):
            status = pytest.main(_arguments(script, args) + ["--collect-only", "-q"], plugins=[recorder])
        listed = [dict(node_id=n, origin_node_id=moved.get(n, (None, None))[0], group=moved.get(n, (None, None))[1])
                  for n in recorder.collected]
        complete = args.keyword is not None or bool(args.case) or set(recorder.collected) == set(moved)
        print(json.dumps(dict(script=script.relative_to(ROOT).as_posix(), origin_revision=origin_revision,
                              cases=listed, complete=complete), indent=1))
        return 0 if status == 0 and complete and all(c["origin_node_id"] for c in listed) else 1
    if not __debug__:
        sys.exit("the cases assert with assert statements, which python -O drops")
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        sys.exit(f"--output {output} is inside the repository {ROOT}")
    if output.exists() and any(output.iterdir()):
        sys.exit(f"--output {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("STARFINDER_WORKFLOW_EVIDENCE", str(output / "workflow-evidence"))
    os.environ.setdefault("STARFINDER_NEXTFLOW_EVIDENCE", str(output / "nextflow-evidence"))
    identity = dict(script=script.relative_to(ROOT).as_posix(), software=_revision(), environment=software(),
                    variables={name: os.environ.get(name) for name in VARIABLES},
                    origin=dict(revision=origin_revision), selection=dict(keyword=args.keyword, cases=args.case),
                    evidence=dict(workflow=os.environ["STARFINDER_WORKFLOW_EVIDENCE"],
                                  nextflow=os.environ["STARFINDER_NEXTFLOW_EVIDENCE"]))
    status = pytest.main(_arguments(script, args, output), plugins=[recorder])
    cases = [dict(node_id=n, origin_node_id=moved.get(n, (None, None))[0], group=moved.get(n, (None, None))[1],
                  **recorder.cases.get(n, dict(outcome="not run", seconds=0.0, message="")))
             for n in recorder.collected]
    (output / "results.json").write_text(json.dumps(dict(identity, pytest_exit_status=int(status), cases=cases),
                                                    indent=1) + "\n", encoding="utf-8")
    outcomes = {case["outcome"] for case in cases}
    print(f"{len(cases)} cases: " + ", ".join(f"{sum(c['outcome'] == o for c in cases)} {o}"
                                               for o in ("passed", "xfailed", "failed", "error", "skipped", "not run")))
    if not cases:
        return 3
    return 1 if outcomes - {"passed", "xfailed", "skipped"} else 2 if "skipped" in outcomes else 0
