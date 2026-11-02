"""W-738: profile resolution, mocked UGER submission and actual job-script execution without UGER."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from . import workflow_tasks as W

pytestmark = [pytest.mark.workflow, pytest.mark.contract]
ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "profile" / "broad-uger"


def _submit_module():
    pytest.importorskip("snakemake", reason="Snakemake not installed")
    spec = importlib.util.spec_from_file_location("broad_submit", PROFILE / "broad-submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("threads,gpu,mem_mb,extra", [
    (1, 1, 8000, ["-l", "h_vmem=8000.0M", "-l", "h_rt=0:30:00",
                   "-l", "gpu=TRUE,operating_system=RedHat8"]),
    (4, 0, 8000, ["-pe", "smp", "4", "-binding", "linear:4", "-R", "y",
                   "-l", "h_vmem=2000.0M", "-l", "h_rt=0:30:00"]),
    (1, 0, 8000, ["-l", "h_vmem=8000.0M", "-l", "h_rt=0:30:00"]),
    (1, 0, 16384, ["-R", "y", "-l", "h_vmem=16384.0M", "-l", "h_rt=0:30:00"]),
])
def test_qsub_lines_for_gpu_four_threads_one_thread_and_high_memory(tmp_path, monkeypatch, capsys,
                                                                    threads, gpu, mem_mb, extra):
    module = _submit_module()
    for name in ("STARFINDER_IMAGE", "STARFINDER_SIF", "APPTAINER_CACHEDIR", "STARFINDER_WEIGHTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    manifest = tmp_path / "output with spaces" / "workflow" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"roots": {"output": str(manifest.parent.parent)},
                                    "tasks": [{"id": "fov/sample/F1", "execution": {"container": None}}]}))
    properties = {"rule": "fov_0", "jobid": 7, "threads": threads,
                  "resources": {"mem_mb": mem_mb, "runtime": 30, "gpu": gpu},
                  "params": {"command": shlex.join(["starfinder", "workflow", "fov", "--manifest", str(manifest),
                                                     "--task", "fov/sample/F1"])}}
    jobscript = tmp_path / "snakejob.fov_0.7.sh"
    jobscript.write_text("#!/bin/bash\n# properties = " + json.dumps(properties) + "\n")
    monkeypatch.setattr(sys, "argv", [str(PROFILE / "broad-submit.py"), "41", "42", str(jobscript)])
    calls = []

    def qsub(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(module.subprocess, "run", qsub)
    module.main()
    assert capsys.readouterr().out == "12345\n"
    logs = manifest.parent / "logs" / "engine"
    assert logs.is_dir()
    exported = (f"STARFINDER_IMAGE={module.CPU_IMAGE},"
                f"STARFINDER_SIF={module.CONTAINERS}/starfinder-{module.CPU_IMAGE[7:]}.sif,"
                f"STARFINDER_THREADS={threads},STARFINDER_GPU={gpu}")
    expected = ["qsub", "-terse", "-N", "fov_0-7", "-r", "y", "-cwd", "-S", "/bin/bash", "-P", "broad",
                "-o", str(logs), "-j", "y", *extra, "-hold_jid", "41,42", "-v", exported, str(jobscript)]
    assert calls == [(expected, {"stdout": subprocess.PIPE, "check": True, "encoding": "utf-8"})]
    print(shlex.join(expected))


def test_the_submit_context_uses_the_manifest_image(tmp_path):
    module = _submit_module()
    image = "sha256:" + "a" * 64
    manifest = tmp_path / "workflow" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"roots": {"output": str(tmp_path)},
                                    "tasks": [{"id": "segment/s/F1/nucleus", "execution": {"container": image}}]}))
    job = {"params": {"command": shlex.join(["starfinder", "workflow", "segment", "--manifest", str(manifest),
                                            "--task", "segment/s/F1/nucleus"])}}
    assert module._context(job) == (tmp_path, image)


# This is an executable stand-in, not a mocked subprocess: it records argv, applies the explicit
# container environment and runs the real command on stdin. It emulates no isolation or GPU runtime.
APPTAINER = '''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
Path(os.environ["CALLS"]).write_text(json.dumps(sys.argv[1:]))
a = sys.argv[1:]
assert a.pop(0) == "exec"
env = {"PATH": os.environ["PATH"]}
while a[0].startswith("--"):
    option = a.pop(0)
    if option == "--env":
        key, value = a.pop(0).split("=", 1)
        env[key] = value
    elif option == "--bind":
        assert a.pop(0) == "/stanley,/local/scratch"
    else:
        assert option in ("--cleanenv", "--nv")
assert a.pop(0).endswith(".sif")
sys.exit(subprocess.run(a, env=env).returncode)
'''


@pytest.mark.parametrize("gpu,busy,exit_code", [(False, False, 0), (True, False, 0),
                                                (True, True, 1), (False, False, 17)])
def test_the_apptainer_job_script_runs_and_preserves_failure(tmp_path, gpu, busy, exit_code):
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "apptainer").write_text(APPTAINER)
    (tools / "apptainer").chmod(0o755)
    (tools / "nvidia-smi").write_text("#!/bin/sh\nprintf '0, 300\n1, " + ("200" if busy else "0") + "\n'\n")
    (tools / "nvidia-smi").chmod(0o755)
    result = tmp_path / "command result.json"
    command = shlex.join([sys.executable, "-c",
                          "import json,os,sys; from pathlib import Path; "
                          "Path(sys.argv[1]).write_text(json.dumps(dict(os.environ))); "
                          "print(\"a small command's output\"); sys.exit(int(sys.argv[2]))",
                          str(result), str(exit_code)])
    script = tmp_path / "job.sh"
    script.write_text((PROFILE / "broad-jobscript.sh").read_text().format(properties="{}", exec_job=command))
    environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                       CALLS=str(tmp_path / "apptainer.json"), TMPDIR=str(tmp_path / "scratch"),
                       APPTAINER_CACHEDIR=str(tmp_path / "cache"), JOB_ID="12345", STARFINDER_THREADS="4",
                       STARFINDER_GPU=str(int(gpu)), STARFINDER_IMAGE="sha256:" + "b" * 64,
                       STARFINDER_SIF=str(tmp_path / "image with spaces.sif"),
                       STARFINDER_WEIGHTS_DIR="/stanley/weights with spaces", HOST_ONLY_VALUE="must disappear")
    done = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True, timeout=30)
    assert done.returncode == exit_code, done.stdout + done.stderr
    if busy:
        assert "No GPU with zero used memory" in done.stderr
        assert not result.exists() and not (tmp_path / "apptainer.json").exists()
        return
    arguments = json.loads((tmp_path / "apptainer.json").read_text())
    assert ("--nv" in arguments) == gpu
    inside = json.loads(result.read_text())
    assert inside["CUDA_VISIBLE_DEVICES"] == ("1" if gpu else "")
    assert inside["STARFINDER_IMAGE"] == environment["STARFINDER_IMAGE"]
    assert inside["JOB_ID"] == "12345"
    assert inside["STARFINDER_WEIGHTS_DIR"] == "/stanley/weights with spaces"
    assert inside["TMPDIR"] == str(tmp_path / "scratch")
    assert all(inside[name] == "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                               "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMEXPR_NUM_THREADS"))
    assert "HOST_ONLY_VALUE" not in inside
    assert done.stdout == "a small command's output\n"
    print("Stand-in Apptainer argv:", json.dumps(arguments))
    print(done.stdout, end="")


def test_the_job_script_refuses_without_a_cache_and_a_lab_folder(tmp_path):
    # W-766: the default cache is /stanley/WangLab/<user>/.apptainer; with no such folder the job stops.
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "apptainer").write_text(APPTAINER)
    (tools / "apptainer").chmod(0o755)
    script = tmp_path / "job.sh"
    script.write_text((PROFILE / "broad-jobscript.sh").read_text().format(properties="{}", exec_job="true"))
    environment = {k: v for k, v in os.environ.items() if k != "APPTAINER_CACHEDIR"}
    environment.update(PATH=str(tools) + os.pathsep + os.environ["PATH"], CALLS=str(tmp_path / "apptainer.json"),
                       TMPDIR=str(tmp_path / "scratch"), JOB_ID="12345", STARFINDER_THREADS="1", STARFINDER_GPU="0",
                       STARFINDER_IMAGE="sha256:" + "b" * 64, STARFINDER_SIF=str(tmp_path / "image.sif"),
                       USER="w766-no-such-user")
    done = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True, timeout=30)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "/stanley/WangLab/w766-no-such-user does not exist; set APPTAINER_CACHEDIR" in done.stderr
    assert not (tmp_path / "apptainer.json").exists()


def test_profile_paths_and_rerun_triggers_resolve_outside_the_repository(tmp_path):
    pytest.importorskip("snakemake", reason="Snakemake not installed")
    pytest.importorskip("snakemake_executor_plugin_cluster_generic", reason="cluster-generic executor not installed")
    config = W.dataset_keys(tmp_path, n_rounds=1, processing={"registration": {"steps": [{"method": "translation"}]}})
    W.write_inputs(config, "sample,fov_id\nsample,F1\n")
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(config))
    # Inspect the settings resolved by the actual CLI and dry-run the maintained dispatcher/rules.
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text('''
import sys
from snakemake.cli import parse_args
# A dry run uses the dryrun executor, so inspect cluster-generic's parsed CLI settings.
_, profile_args = parse_args(sys.argv[1:])
print("PROFILE_SUBMIT=" + profile_args.cluster_generic_submit_cmd)
print("PROFILE_STATUS=" + profile_args.cluster_generic_status_cmd)
print("PROFILE_JOBSCRIPT=" + str(profile_args.jobscript))
print("PROFILE_RERUN_TRIGGERS=" + ",".join(sorted(t.name.lower() for t in profile_args.rerun_triggers)))
include: ''' + repr(str(ROOT / "workflow" / "Snakefile")) + "\n")
    environment = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    command = [str(Path(sys.executable).with_name("snakemake")), "--profile", str(PROFILE), "--snakefile",
               str(snakefile), "--configfile", str(configuration), "--dry-run", "--nocolor", "--printshellcmds",
               "--", "all"]
    done = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
    print("Working directory:", tmp_path)
    print("Command:", shlex.join(command))
    print(done.stdout + done.stderr)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PROFILE_SUBMIT=" + str(PROFILE / "broad-submit.py") in done.stdout
    assert "PROFILE_STATUS=" + str(PROFILE / "broad-status.py") in done.stdout
    assert "PROFILE_JOBSCRIPT=" + str(PROFILE / "broad-jobscript.sh") in done.stdout
    assert "PROFILE_RERUN_TRIGGERS=input,mtime,params" in done.stdout
    assert "rule fov_0:" in done.stdout and "--task fov/sample/F1" in done.stdout
