"""W-787, W-805: the ``sge`` profile's container options for a CPU and a GPU task, and a task record's job id, rendered by
Nextflow without UGER; W-796: a fov task of a learned detector on cuda runs as a GPU job."""
from __future__ import annotations

import copy
import json
import os
import re
import shlex
import stat
import sys
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

#: The scheduler's job id the stand-in ``qsub`` gives the job script (UGER's `segment` job of W-795 was 62751475).
JOB_ID = "62751475"
#: The scheduler's job id, passed into the container by the job script's shell (W-805).
JOB = "--env JOB_ID=$JOB_ID"
#: The options at the base revision (e78f755, before W-787): the thread variables and the caches, with ``--nv`` first
#: for a GPU task, and W-805's job id after them. With ``weights_dir`` unset the rendered options must stay exactly these.
BASE_OPTIONS = {"cpu": " ".join(f"--env {name}=1" for name in THREADS) + " " + CACHES + " " + JOB,
                "gpu": "--nv " + " ".join(f"--env {name}=1" for name in THREADS) + " " + CACHES + " " + JOB}

# Stand-ins for the scheduler and the container runtime, which the render needs only to start each task. ``qsub``
# runs the job script `.command.run` that Nextflow wrote and so reaches the ``apptainer`` line, which records its
# arguments, as the shell of the job script passed them, and runs the task's own command, with the ``--env`` pairs
# it was given, outside any container.
QSUB = f'''#!/bin/bash
export JOB_ID={JOB_ID}
bash "${{@: -1}}" > /dev/null 2>&1
echo 'Your job {JOB_ID} ("stand-in") has been submitted'
'''
QSTAT = "#!/bin/bash\nexit 0\n"
APPTAINER = '''#!/bin/bash
mkdir -p "$(dirname "$0")/argv"
printf '%s\\0' "$@" > "$(dirname "$0")/argv/${APPTAINERENV_NXF_TASK_WORKDIR//\\//_}"
shift
pairs=()
while [[ "$1" == -* ]]; do
  case "$1" in --env) pairs+=("$2"); shift 2;; --bind|-B) shift 2;; *) shift;; esac
done
shift
for name in $(compgen -e APPTAINERENV_); do pairs+=("${name#APPTAINERENV_}=${!name}"); done
# As the container starts: nothing of the job script's environment but PATH, the --env pairs and the APPTAINERENV_ ones.
exec env -i PATH="$PATH" "${pairs[@]}" "$@"
'''
# The task commands of the render that does not stub: the plan and the verification are this checkout's package (the
# job script clears the environment, PYTHONPATH with it); a task's command writes the record Nextflow copies, holding
# the execution context it was given.
STARFINDER = f'''#!{sys.executable}
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
if argv[1] in ("plan", "verify"):
    if "--final" in argv:
        sys.exit(0)
    os.execve(sys.executable, [sys.executable, "-m", "starfinder", *argv],
              dict(os.environ, PYTHONPATH={str(X.ROOT / "src" / "python")!r}))
option = lambda name: argv[argv.index(name) + 1]
manifest = json.loads(Path(option("--manifest")).read_text())
task = next(t for t in manifest["tasks"] if t["id"] == option("--task"))
record = Path(manifest["roots"]["output"]) / task["record"]
record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps({{"status": "succeeded", "workflow": {{"execution_context": {{
    "engine": option("--engine"), "job_id": option("--job-id"), "attempt": int(option("--attempt")),
    "image": option("--image")}}}}}}) + "\\n")
'''


def stand_ins(folder):
    folder.mkdir(parents=True)
    for name, text in (("qsub", QSUB), ("qstat", QSTAT), ("apptainer", APPTAINER), ("starfinder", STARFINDER)):
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


def render(tmp_path_factory, *arguments, stub=True):
    """Run the morphology-only variant (a CPU morphology task and a CUDA StarDist task) under ``sge``: as stubs, or, with
    ``stub=False``, with the stand-in ``starfinder`` as every task's command. Returns the `apptainer exec` lines, or,
    for ``stub=False``, the work folder and the manifest."""
    N.executable()
    work = tmp_path_factory.mktemp("sge-render")
    X.golden().write_fixture(work)
    config = N.small_resources(X.variants(work)["morphology-only"])
    path = X.write_config(work, "sge", config)
    tools = stand_ins(work / "tools")
    done, rows = N.run(work, path, "--container", str(work / "image.sif"), "-work-dir", str(work / "work"),
                       *arguments, environment={"PATH": str(tools) + os.pathsep + os.environ["PATH"],
                                                "NXF_OFFLINE": "true", "NXF_PLUGINS_DEFAULT": "false"},
                       profile="sge", stub=stub)
    assert done.returncode == 0, done.stdout + done.stderr
    assert {r["status"] for r in N.tasks(rows)} == {"COMPLETED"}
    return apptainer_lines(work / "work", tools) if stub else (work / "work", N.manifest(path))


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


def task_records(work, manifest):
    """{task id: (its record, the `.command.sh` that wrote it)}: the record is the one copied into the task's work folder."""
    found = {}
    for task in manifest["tasks"]:
        record = task["id"].replace("/", ".") + ".json"
        (script,) = [s for s in Path(work).glob("*/*/.command.sh") if f"'--task' '{task['id']}'" in s.read_text()]
        found[task["id"]] = (json.loads((script.parent / record).read_text()), script)
    return found


@pytest.fixture(scope="module")
def sge_tasks(tmp_path_factory):
    work, manifest = render(tmp_path_factory, stub=False)
    return manifest, task_records(work, manifest)


def test_under_sge_each_record_holds_the_schedulers_job_id(sge_tasks):
    """The stand-in ``qsub`` sets JOB_ID before it runs `.command.run`; the stand-in ``apptainer`` starts the task with
    nothing of that environment but the --env pairs, so the id reaches the record only through the profile's pair (W-805)."""
    manifest, records = sge_tasks
    assert set(records) == {t["id"] for t in manifest["tasks"]}
    assert {t["execution"]["gpu"] for t in manifest["tasks"]} == {True, False}
    for task_id, (record, script) in records.items():
        assert record["workflow"]["execution_context"]["job_id"] == JOB_ID, task_id
        assert '--job-id "${JOB_ID:-$task_hash}"' in script.read_text(), task_id


def test_under_sge_each_task_is_given_the_image(sge_tasks):
    """W-831: the image the engine gives a task reaches its command as ``--image`` (the record's ``image``)."""
    manifest, records = sge_tasks
    images = {r["workflow"]["execution_context"]["image"] for r, _ in records.values()}
    assert len(images) == 1 and images.pop().endswith("/image.sif")


def test_on_the_host_each_task_is_given_no_image(local_tasks):
    """W-831: with no container engine, the command's ``--image`` is ``none``, which the record stores as null."""
    manifest, records = local_tasks
    assert {r["workflow"]["execution_context"]["image"] for r, _ in records.values()} == {"none"}


def test_the_fallback_is_the_task_hash_from_its_work_folder(local_tasks):
    manifest, records = local_tasks
    assert set(records) == {t["id"] for t in manifest["tasks"]}
    for task_id, (record, script) in records.items():
        job_id = record["workflow"]["execution_context"]["job_id"]
        assert job_id == script.parent.parent.name + script.parent.name, task_id
        assert re.fullmatch(r"[0-9a-f]{32}", job_id), task_id
    assert len({r["workflow"]["execution_context"]["job_id"] for r, _ in records.values()}) == len(records)


@pytest.fixture(scope="module")
def local_tasks(tmp_path_factory):
    """The same variant under ``local``: no scheduler, so no JOB_ID, and the command is the stand-in ``starfinder``."""
    N.executable()
    work = tmp_path_factory.mktemp("local-render")
    X.golden().write_fixture(work)
    config = N.small_resources(X.variants(work)["morphology-only"])
    config["segmentation"]["device"] = "cpu"  # ``local`` has no accelerator
    path = X.write_config(work, "local", config)
    tools = stand_ins(work / "tools")
    done, rows = N.run(work, path, "-work-dir", str(work / "work"), stub=False,
                       environment={"PATH": str(tools) + os.pathsep + os.environ["PATH"], "NXF_OFFLINE": "true",
                                    "NXF_PLUGINS_DEFAULT": "false"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert {r["status"] for r in N.tasks(rows)} == {"COMPLETED"}
    manifest = N.manifest(path)
    return manifest, task_records(work / "work", manifest)


def test_a_fov_task_of_a_learned_detector_on_cuda_runs_as_a_gpu_job(tmp_path_factory):
    """W-796: main.nf routes a fov task whose ``execution.gpu`` is true (Spotiflow on cuda) to FOV_GPU, whose label
    ``gpu`` gives its job script the GPU request and ``--nv``; with segmentation on the CPU, no other task does."""
    N.executable()
    work = tmp_path_factory.mktemp("sge-fov-gpu")
    X.golden().write_fixture(work)
    config = X.variants(work)["morphology-only"]
    config["segmentation"]["device"] = "cpu"
    processing = copy.deepcopy(X.PROCESSING)
    processing["spot_finding"] = {"method": "spotiflow", "model": "smfish_3d", "device": "cuda"}
    path = X.write_config(work, "sge-fov-gpu", N.small_resources(dict(config, processing=processing)))
    tools = stand_ins(work / "tools")
    done, rows = N.run(work, path, "--container", str(work / "image.sif"), "-work-dir", str(work / "work"),
                       environment={"PATH": str(tools) + os.pathsep + os.environ["PATH"], "NXF_OFFLINE": "true",
                                    "NXF_PLUGINS_DEFAULT": "false"}, profile="sge")
    assert done.returncode == 0, done.stdout + done.stderr
    processes = {r["tag"]: r["name"].split(" ")[0] for r in N.tasks(rows)}
    tasks = N.manifest(path)["tasks"]
    jobs = {}
    for script in Path(work / "work").glob("*/*/.command.run"):
        # The stub's command names the task as one quoted word (TaskCommands.quote); PLAN and VERIFY name none.
        named = [t["id"] for t in tasks if f"'{t['id']}'" in (script.parent / ".command.sh").read_text()]
        if not named:
            continue
        (task_id,) = named
        text = script.read_text()
        (line,) = [line for line in text.splitlines() if "apptainer exec" in line]
        jobs[task_id] = ("-l gpu=TRUE,operating_system=RedHat8" in text, "--nv" in line.split())
    assert {t["kind"] for t in tasks} >= {"fov", "morphology", "segment"} and set(jobs) == set(processes)
    for task in tasks:
        gpu = task["kind"] == "fov"
        assert task["execution"]["gpu"] is gpu, task["id"]
        assert processes[task["id"]] == ("FOV_GPU" if gpu else processes[task["id"]].removesuffix("_GPU")), task["id"]
        assert jobs[task["id"]] == (gpu, gpu), task["id"]
