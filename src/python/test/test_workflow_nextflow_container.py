"""W-831: with a container engine on, Nextflow starts only when an image resolves for the tasks.

A ``process.container`` in a ``-c`` file never reaches the workflow's processes (the labelled selector of
``nextflow.config`` outranks it), so an engine enabled there left every task on the host. The head now stops before
PLAN, naming the engine, unless ``--container`` or the configuration's ``execution.container`` gives an image. The runs
are stubs; a stand-in for each engine runs the task's command where it stands, so no container starts and no real engine is called.
"""
from __future__ import annotations

import os
import stat

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]

IMAGE = "registry.example/starfinder@sha256:" + "0" * 64
ENGINES = ["docker", "apptainer", "singularity"]

# What Nextflow's job script calls: ``docker run <options> <image> /bin/bash -ue .command.sh`` (then ``docker rm``), or
# ``apptainer exec`` / ``singularity exec`` with the same tail. The stand-in, one script under each engine's name, records
# each call and runs the command after the image on the host. No test calls a real engine: a real one would pull
# ``IMAGE`` from a registry.
ENGINE_STAND_IN = '''#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$(dirname "$0")/calls"
if [[ "$1" == run || "$1" == exec ]]; then
  shift
  while [[ $# -gt 0 && "$1" != /bin/bash ]]; do shift; done
  exec "$@"
fi
exit 0
'''


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    N.executable()
    work = tmp_path_factory.mktemp("nextflow-container")
    X.golden().write_fixture(work)
    return work


def configuration(work, name, container=None):
    settings = N.small_resources(X.variants(work)["morphology-only"])
    settings["output_id"] = "container-" + name
    if container:
        settings["execution"]["container"] = container
    return X.write_config(work, name, settings)


def engine_file(work, name, engine):
    path = work / f"{name}.config"
    path.write_text(f"{engine}.enabled = true\nprocess.container = {IMAGE!r}\n")
    return path


def engine_stand_ins(work, name):
    """A folder for ``name`` with a stand-in ``docker``, ``apptainer`` and ``singularity``, for the front of ``PATH``.
    Its file ``calls`` lists every call made to any of them. Returns the folder and the environment that puts it first."""
    folder = work / f"tools-{name}"
    folder.mkdir()
    for engine in ENGINES:
        (folder / engine).write_text(ENGINE_STAND_IN)
        (folder / engine).chmod((folder / engine).stat().st_mode | stat.S_IXUSR)
    (folder / "calls").write_text("")
    return folder, {"PATH": str(folder) + os.pathsep + os.environ["PATH"]}


def run(work, name, *arguments, **keywords):
    return N.run(work / name, *arguments, environment={"NXF_OFFLINE": "true", "NXF_PLUGINS_DEFAULT": "false",
                                                       **keywords.pop("environment", {})}, **keywords)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_in_a_c_file_without_a_container_stops_before_plan(tiny, engine):
    name = "refuse-" + engine
    config = configuration(tiny, name)
    tools, environment = engine_stand_ins(tiny, name)
    done, rows = run(tiny, name, config, "-c", str(engine_file(tiny, name, engine)), environment=environment)
    text = done.stdout + done.stderr
    assert done.returncode != 0, text
    assert f"The {engine} engine is enabled but no container image is set" in text, text
    assert "--container <image>" in text and "execution.container" in text, text
    assert rows == [], "PLAN or a task started"
    assert (tools / "calls").read_text() == "", "the engine was called"


def test_the_container_option_passes_the_check(tiny):
    name = "option"
    config = configuration(tiny, name)
    tools, environment = engine_stand_ins(tiny, name)
    done, rows = run(tiny, name, config, "--container", IMAGE, "-c", str(engine_file(tiny, name, "docker")),
                     environment=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no container image is set" not in done.stdout + done.stderr
    assert {r["name"].split(" (")[0] for r in rows} >= {"PLAN", "VERIFY_BEFORE", "VERIFY_AFTER"}
    assert {r["status"] for r in rows} == {"COMPLETED"}
    # PLAN is the one process that must have run in the image, which is the option's own image.
    assert IMAGE in (tools / "calls").read_text()


def test_the_configurations_execution_container_passes_the_check(tiny):
    name = "execution"
    config = configuration(tiny, name, container=IMAGE)
    tools, environment = engine_stand_ins(tiny, name)
    done, rows = run(tiny, name, config, "-c", str(engine_file(tiny, name, "docker")), environment=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no container image is set" not in done.stdout + done.stderr
    assert {r["status"] for r in N.tasks(rows)} == {"COMPLETED"}
    assert IMAGE in (tools / "calls").read_text()


def test_without_an_engine_nothing_changes(tiny):
    name = "host"
    config = configuration(tiny, name)
    tools, environment = engine_stand_ins(tiny, name)
    done, rows = run(tiny, name, config, environment=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no container image is set" not in done.stdout + done.stderr
    assert N.tasks(rows) and {r["status"] for r in rows} == {"COMPLETED"}
    # A -c file that sets only process.container, with no engine, is still no refusal.
    other = tiny / "no-engine.config"
    other.write_text(f"process.container = {IMAGE!r}\n")
    done, rows = run(tiny, name + "-c", configuration(tiny, name + "-c"), "-c", str(other), environment=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tools / "calls").read_text() == "", "an engine was called with none enabled"
