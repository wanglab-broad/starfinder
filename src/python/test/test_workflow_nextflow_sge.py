"""W-787: the ``sge`` profile's container options for a CPU and a GPU task, rendered by Nextflow without UGER."""
from __future__ import annotations

import os
import re
import shlex
import stat
from pathlib import Path

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
           "NUMEXPR_NUM_THREADS")
CACHES = "--env XDG_CACHE_HOME=$NXF_TASK_WORKDIR/.cache --env MPLCONFIGDIR=$NXF_TASK_WORKDIR/.config/matplotlib"
WEIGHTS = "/stanley/WangLab/weights"
#: Folders a shell would split or unquote if the profile inserted them as they are.
FOLDERS = [WEIGHTS, "/stanley/model weights", "/stanley/it's a 'model' folder"]

#: The options at the base revision (e78f755, before W-787): the thread variables and the caches, with ``--nv`` first
#: for a GPU task. With ``weights_dir`` unset the rendered options must stay exactly these.
BASE_OPTIONS = {"cpu": " ".join(f"--env {name}=1" for name in THREADS) + " " + CACHES,
                "gpu": "--nv " + " ".join(f"--env {name}=1" for name in THREADS) + " " + CACHES}

# Stand-ins for the scheduler and the container runtime, which the render needs only to start each task. ``qsub``
# runs the job script `.command.run` that Nextflow wrote and so reaches the ``apptainer`` line, which records its
# arguments, as the shell of the job script passed them, and runs the task's own command, with the ``--env`` pairs
# it was given, outside any container.
QSUB = '''#!/bin/bash
bash "${@: -1}" > /dev/null 2>&1
echo 'Your job 1 ("stand-in") has been submitted'
'''
QSTAT = "#!/bin/bash\nexit 0\n"
APPTAINER = '''#!/bin/bash
mkdir -p "$(dirname "$0")/argv"
printf '%s\\0' "$@" > "$(dirname "$0")/argv/${APPTAINERENV_NXF_TASK_WORKDIR//\\//_}"
shift
while [[ "$1" == -* ]]; do
  case "$1" in --env) export "$2"; shift 2;; --bind|-B) shift 2;; *) shift;; esac
done
shift
for name in $(compgen -e APPTAINERENV_); do export "${name#APPTAINERENV_}=${!name}"; done
exec "$@"
'''


def stand_ins(folder):
    folder.mkdir(parents=True)
    for name, text in (("qsub", QSUB), ("qstat", QSTAT), ("apptainer", APPTAINER)):
        (folder / name).write_text(text)
        (folder / name).chmod(folder.joinpath(name).stat().st_mode | stat.S_IXUSR)
    return folder


def apptainer_lines(work, tools):
    """{label: (the `apptainer exec` line of a task's `.command.run`, the arguments the stand-in received)} for one CPU
    task and the GPU task."""
    found = {}
    for script in sorted(Path(work).glob("*/*/.command.run")):
        lines = [line for line in script.read_text().splitlines() if "apptainer exec" in line]
        assert len(lines) == 1, script
        argv = (tools / "argv" / str(script.parent).replace("/", "_")).read_text().split("\0")[:-1]
        found.setdefault("gpu" if " --nv " in lines[0] else "cpu", (lines[0], argv))
    return found


def render(tmp_path_factory, *arguments):
    """Run the morphology-only variant (a CPU morphology task and a CUDA StarDist task) as stubs under ``sge``."""
    N.executable()
    work = tmp_path_factory.mktemp("sge-render")
    X.golden().write_fixture(work)
    config = N.small_resources(X.variants(work)["morphology-only"])
    path = X.write_config(work, "sge", config)
    tools = stand_ins(work / "tools")
    done, rows = N.run(work, path, "--container", str(work / "image.sif"), "-work-dir", str(work / "work"),
                       *arguments, environment={"PATH": str(tools) + os.pathsep + os.environ["PATH"],
                                                "NXF_OFFLINE": "true", "NXF_PLUGINS_DEFAULT": "false"},
                       profile="sge")
    assert done.returncode == 0, done.stdout + done.stderr
    assert {r["status"] for r in N.tasks(rows)} == {"COMPLETED"}
    return apptainer_lines(work / "work", tools)


@pytest.fixture(scope="module")
def unset(tmp_path_factory):
    return {label: line for label, (line, _) in render(tmp_path_factory).items()}


@pytest.fixture(scope="module", params=FOLDERS, ids=["plain", "spaces", "quotes"])
def given(request, tmp_path_factory):
    return request.param, render(tmp_path_factory, "--weights_dir", request.param)


def test_each_label_runs_in_the_rendered_job_script(unset, given):
    assert set(unset) == set(given[1]) == {"cpu", "gpu"}


@pytest.mark.parametrize("label", ["cpu", "gpu"])
def test_weights_dir_is_passed_to_apptainer_for_cpu_and_gpu_tasks(given, label):
    folder, rendered = given
    line, argv = rendered[label]
    words = shlex.split(line)
    pair = f"STARFINDER_WEIGHTS_DIR={folder}"
    # The folder is one argument of the apptainer command, as the line of the job script reads and as the shell passed it.
    for arguments in (words, argv):
        assert [w for w in arguments if "STARFINDER_WEIGHTS_DIR" in w] == [pair]
        assert arguments[arguments.index(pair) - 1] == "--env"
    # The rest of the options are the base revision's, in their order, with the new pair after the caches.
    base = shlex.split(BASE_OPTIONS[label])
    end = words.index(pair) - 1
    assert words[end - len(base):end] == base


@pytest.mark.parametrize("label", ["cpu", "gpu"])
def test_without_weights_dir_the_options_are_the_base_revisions(unset, label):
    line = unset[label]
    assert "STARFINDER_WEIGHTS_DIR" not in line
    assert re.search(r"apptainer exec .*?" + re.escape(BASE_OPTIONS[label]) + r" \S+\.sif ", line), line
    assert ("--nv" in line.split()) == (label == "gpu")


def test_the_parameter_is_declared_and_unset_by_default():
    text = (X.ROOT / "nextflow" / "nextflow.config").read_text()
    assert re.search(r"^\s+weights_dir = null$", text, re.MULTILINE)
