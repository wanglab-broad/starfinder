"""C4 and C9's MATLAB dry run of design/validation/orchestration.md: real Snakemake runs of ``workflow/Snakefile``.

C4: the dry run of each worked-example variant on ``tiny_sample`` schedules each task of the plan's manifest once, by
the rule of its group (``rules/tasks.smk``: one rule per rule group, named ``<kind>_<group index>``), and no other
job; a refused variant schedules no job and stops with the plan's message (exit 2 of the plan). After a completed
tiled run of ``exactness_fov`` a second dry run schedules nothing, although the run removed the temporary chunked
checkpoint (open point E-1). After a completed ``per-fov`` run, a dry run schedules: nothing; after a changed
``processing`` key, the tasks whose kind reads that block and their downstream; after a changed ``export`` key, those of
the export block and their downstream (``export``, ``report`` and ``compare``: ``compare`` needs ``report``, and a
fingerprint holds those of the task's needs); after a raw image replaced at the same path with an older modification
time, the tasks that declare it and their downstream (Snakemake's params trigger on each task's fingerprint; the mtime
trigger alone misses an older time). The rerun triggers are the engine record's profile setting, ``mtime``, ``params``
and ``input``. C9: the MATLAB dry run of ``tests/tissue_2D_test.yaml``, adapted as W-641 adapted it (one FOV, paths
under the test's folder, its input folders and sample annotation written) with ``backend: matlab``, which a
configuration needs for the dispatcher to include ``rules/legacy.smk``, schedules W-641's three jobs.

Snakemake 9 comes from the ``workflow`` dependency group, beside the interpreter; the tests skip with the reason
"Snakemake not installed" where it is absent. The BigStitcher engine task's job is checked by its printed
shell command: Fiji from its tool command, then ``bigstitcher-check`` (no Fiji runs). Each Snakemake process runs with one numerical thread and the
``starfinder`` command of this environment first on PATH.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from starfinder._execution import THREAD_VARIABLES
from starfinder.workflow import TASK_BLOCKS

from . import tiled_fixtures as T
from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.slow]

SNAKEFILE = X.ROOT / "workflow" / "Snakefile"
TRIGGERS = ["--rerun-triggers", "mtime", "params", "input"]
REFUSALS = {"refused-assignment": "add a processing block", "refused-matlab": "add backend: matlab"}


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


def scheduled(done):
    """(jobs per rule without ``all``, the task ids of the scheduled jobs' commands) of a dry run."""
    assert done.returncode == 0, done.stdout + done.stderr
    if "Nothing to be done" in done.stdout:
        return {}, Counter()
    table = done.stdout[done.stdout.index("Job stats:"):].split("\n\n")[0]
    jobs = {rule: int(count) for rule, count in re.findall(r"^(\w+)\s+(\d+)$", table, re.M)
            if rule not in ("all", "total")}
    return jobs, Counter(re.findall(r" --task (\S+) ", done.stdout))


def manifest(output_root):
    return json.loads((Path(output_root) / "workflow" / "manifest.json").read_text())


def downstream(plan, tasks):
    """``tasks`` and every task that needs one of them, transitively (the manifest's order respects needs)."""
    found = set(tasks)
    for task in plan["tasks"]:
        if found.intersection(task["needs"]):
            found.add(task["id"])
    return found


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    _snakemake()
    work = tmp_path_factory.mktemp("dryrun") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


@pytest.mark.parametrize("variant", X.VARIANTS + X.REFUSED)
def test_the_dry_run_of_each_variant_schedules_the_manifests_tasks_per_rule_group(tiny, variant):
    config = X.write_config(tiny, variant, X.variants(tiny)[variant])
    done = snakemake(tiny, config, "--dry-run")
    if variant in X.REFUSED:
        assert done.returncode != 0
        message = done.stdout + done.stderr
        assert "exited with status 2" in message and REFUSALS[variant] in message, message
        assert "Job stats" not in done.stdout
        return
    jobs, tasks = scheduled(done)
    plan = manifest(X.output_root(tiny, f"example-{variant}"))
    per_group = Counter(task["group"] for task in plan["tasks"])
    assert jobs == {f"{group['kind']}_{index}": per_group[group["name"]]
                    for index, group in enumerate(plan["groups"])}
    assert tasks == Counter(task["id"] for task in plan["tasks"])
    kinds = Counter(task["kind"] for task in plan["tasks"])
    assert dict(kinds) == X.KINDS[variant]
    for task in plan["tasks"]:
        if "tool_command" in task:
            # The BigStitcher engine task (D3): its declared output removed, Fiji from the tool command with the
            # configured path, then bigstitcher-check, which runs whatever Fiji's exit status and decides.
            tool = [task["execution"]["tool"]["path"] if part == "{fiji}" else part for part in task["tool_command"]]
            shell = (f"rm -f {task['outputs'][0]}; {' '.join(tool)} || echo \"the tool exited with status $?\" >&2; "
                     f"{' '.join(task['command'])} --engine snakemake --engine-version ")
            assert shell in done.stdout, done.stdout


def test_a_dry_run_after_a_completed_tiled_run_schedules_nothing(tmp_path):
    """E-1: the merge's removal of the temporary chunked checkpoint, a temp() output of tile_global, leaves the run
    complete for Snakemake."""
    _snakemake()
    name = T.write_exactness_fov(tmp_path / "input")
    (tmp_path / "input" / name / T.SAMPLE / "fovs.csv").write_text(f"sample,fov_id\n{T.SAMPLE},{T.FOV}\n")
    config = T.configuration(tmp_path / "input", tmp_path / "output", name, mode="tiled",
                             tiled={"tile_size_yx": 32, "halo": 6})
    path = tmp_path / "tiled.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    done = snakemake(tmp_path, path, *TRIGGERS)
    assert done.returncode == 0, done.stdout + done.stderr
    out = tmp_path / "output" / name / "run"
    kinds = Counter(task["kind"] for task in manifest(out)["tasks"])
    assert kinds == {"tile_global": 1, "tile": 4, "tile_merge": 1}
    assert not (out / "fovs" / T.FOV / "tiled" / "registered.h5").exists()
    assert (out / "fovs" / T.FOV / "molecules.csv").is_file()
    assert scheduled(snakemake(tmp_path, path, "--dry-run", *TRIGGERS)) == ({}, Counter())


@pytest.fixture(scope="module")
def per_fov_run(tmp_path_factory):
    """(work, configuration, its file, output root) after a completed Snakemake run of the per-fov variant, on a
    fixture of its own (a test replaces one of its raw images)."""
    _snakemake()
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("the per-fov variant's stitching and export need the multiview-stitcher and anndata extras")
    work = tmp_path_factory.mktemp("per-fov") / "work"
    X.golden().write_fixture(work)
    config = X.variants(work)["per-fov"]
    path = X.write_config(work, "per-fov", config)
    done = snakemake(work, path, *TRIGGERS)
    assert done.returncode == 0, done.stdout[-5000:] + done.stderr[-5000:]
    return work, config, path, X.output_root(work, "example-per-fov")


def _changed(per_fov_run, change):
    """The task ids a dry run schedules after ``change(configuration)`` (the configuration is restored after)."""
    work, config, path, _ = per_fov_run
    changed = copy.deepcopy(config)
    change(changed)
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    try:
        return scheduled(snakemake(work, path, "--dry-run", *TRIGGERS))[1]
    finally:
        path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_a_dry_run_after_a_completed_per_fov_run_schedules_nothing(per_fov_run):
    work, _, path, out = per_fov_run
    assert scheduled(snakemake(work, path, "--dry-run", *TRIGGERS)) == ({}, Counter())
    plan = manifest(out)
    assert all((out / task["record"]).is_file() for task in plan["tasks"])


@pytest.mark.parametrize("block", ["processing", "export"])
def test_a_changed_key_reruns_the_tasks_of_its_block_and_their_downstream(per_fov_run, block):
    changes = {"processing": lambda c: c["processing"]["spot_finding"].update(threshold_value=6.0),
               "export": lambda c: c["export"].update(fuse_labels=True)}
    tasks = _changed(per_fov_run, changes[block])
    plan = manifest(per_fov_run[3])
    readers = {t["id"] for t in plan["tasks"] if block in TASK_BLOCKS[t["kind"]]}
    assert set(tasks) == downstream(plan, readers) and set(tasks.values()) == {1}
    if block == "export":
        assert set(tasks) == {"export/sample", "report/sample", "compare"}
    else:
        assert {t["kind"] for t in plan["tasks"] if t["id"] not in tasks} == {"morphology"}


def test_a_raw_image_replaced_with_an_older_time_reruns_the_tasks_that_declare_it_and_their_downstream(per_fov_run):
    work, _, path, out = per_fov_run
    image = X.input_root(work) / "round2" / "FOV_002" / "ch01.ome.tif"
    stat = image.stat()
    image.write_bytes(image.read_bytes()[::-1])
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns - 86_400 * 10**9))
    tasks = scheduled(snakemake(work, path, "--dry-run", *TRIGGERS))[1]
    plan = manifest(out)
    declaring = {t["id"] for t in plan["tasks"] if "input:round2/FOV_002/ch01.ome.tif" in t["inputs"]}
    assert declaring == {"summarize/sample/percentile_normalization/FOV_002", "fov/sample/FOV_002"}
    assert set(tasks) == downstream(plan, declaring) and set(tasks.values()) == {1}


def test_the_matlab_dry_run_schedules_w641s_three_jobs(tmp_path):
    """C9: tests/tissue_2D_test.yaml through the dispatcher and rules/legacy.smk (backend: matlab)."""
    _snakemake()
    config = yaml.safe_load((X.ROOT / "tests" / "tissue_2D_test.yaml").read_text())
    assert "backend" not in config
    config.update(backend="matlab", config_path=str(tmp_path / "config.yaml"), starfinder_path=str(X.ROOT),
                  envs_path=str(tmp_path / "unavailable-envs_path"), fiji_path=str(tmp_path / "unavailable-fiji_path"),
                  root_input_path=str(tmp_path / "input"), root_output_path=str(tmp_path / "output"), n_fovs=2)
    root = tmp_path / "input" / config["dataset_id"] / config["sample_id"]
    for number in range(1, config["n_rounds"] + 1):
        (root / f"round{number}" / "tile_1").mkdir(parents=True)
    (root / "genes.csv").write_text("")
    annotation = tmp_path / "output" / config["dataset_id"] / config["output_id"] / "documents"
    annotation.mkdir(parents=True)
    (annotation / "sample-annotation.csv").write_text(f"sample_id,fov_start,fov_end\n{config['sample_id']},1,2\n")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    done = snakemake(tmp_path, tmp_path / "config.yaml", "--dry-run", "--", "all")
    assert done.returncode == 0, done.stdout + done.stderr
    table = done.stdout[done.stdout.index("Job stats:"):].split("\n\n")[0]
    assert dict(re.findall(r"^(\w+)\s+(\d+)$", table, re.M)) == {"all": "1", "rsf_preparation": "1",
                                                                  "rsf_single_fov": "1", "total": "3"}
    assert "wildcards: fovID=tile_1" in done.stdout
