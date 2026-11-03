#!/usr/bin/env python3
"""Submit a Python workflow job to Broad UGER; never deploy software on a node."""

import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from snakemake.utils import read_job_properties


# W-792's CPU image (0.1.dev719+g92c723b15, with procps); SIFs are built beside the archives on a Broad node.
CPU_IMAGE = "sha256:ac4604d2955e1532d0ba5ecc208f8299cfd5053feb506544439d6c58433d6ec9"
CONTAINERS = "/stanley/WangLab/jiahao/test/starfinder_benchmark/containers"


def _context(job):
    """Read the task's image and output root from the command's manifest."""
    words = shlex.split(job["params"]["command"])
    manifest_path = Path(words[words.index("--manifest") + 1])
    task_id = words[words.index("--task") + 1]
    manifest = json.loads(manifest_path.read_text())
    task = next(task for task in manifest["tasks"] if task["id"] == task_id)
    image = task["execution"]["container"] or os.environ.get("STARFINDER_IMAGE", CPU_IMAGE)
    if not image.startswith("sha256:") or len(image) != 71 or any(c not in "0123456789abcdef" for c in image[7:]):
        raise ValueError("the UGER image must be an OCI manifest digest (sha256:<64 hex digits>)")
    return Path(manifest["roots"]["output"]), image


def main():
    """Submit the last argument's job script, with any preceding job dependencies."""
    dependencies, jobscript = sys.argv[1:-1], sys.argv[-1]
    job = read_job_properties(jobscript)
    resources = job.get("resources", {})
    cluster = job.get("cluster", {})
    threads = job.get("threads", 1)
    mem_mb = cluster.get("mem_mb", resources.get("mem_mb"))
    runtime = cluster.get("runtime", resources.get("runtime"))
    jobid = str(job["jobid"])
    root, image = _context(job)
    logs = root / "workflow" / "logs" / "engine"
    logs.mkdir(parents=True, exist_ok=True)
    # UGER adds its job ID to this directory's log names, so retries keep their logs.
    command = ["qsub", "-terse", "-N", f"{job['rule']}-{jobid}", "-r", "y", "-cwd", "-S", "/bin/bash",
               "-P", cluster.get("project", "broad"), "-o", str(logs), "-j", "y"]
    if threads > 1:
        command.extend(["-pe", "smp", str(threads), "-binding", f"linear:{threads}"])
    if (mem_mb and mem_mb > 15 * 1024) or threads >= 4:
        command.extend(["-R", "y"])
    if mem_mb:
        command.extend(["-l", f"h_vmem={round(mem_mb / threads, 2)}M"])
    if runtime:
        command.extend(["-l", f"h_rt={datetime.timedelta(minutes=runtime)}"])
    gpu = bool(resources.get("gpu", 0))
    if gpu:
        command.extend(["-l", "gpu=TRUE,operating_system=RedHat8"])
    if dependencies:
        command.extend(["-hold_jid", ",".join(dependencies)])

    # qsub does not inherit arbitrary head-job variables. Export only the container context,
    # not the head's Python paths or software environment, into the clean container.
    environment = {
        "STARFINDER_IMAGE": image,
        "STARFINDER_SIF": os.environ.get("STARFINDER_SIF", f"{CONTAINERS}/starfinder-{image[7:]}.sif"),
        "STARFINDER_THREADS": str(threads),
        "STARFINDER_GPU": str(int(gpu)),
    }
    for name in ("APPTAINER_CACHEDIR", "STARFINDER_WEIGHTS_DIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    if any("," in value or "\n" in value for value in environment.values()):
        raise ValueError("UGER exported container settings cannot contain commas or newlines")
    command.extend(["-v", ",".join(f"{key}={value}" for key, value in environment.items()), jobscript])
    proc = subprocess.run(command, stdout=subprocess.PIPE, check=True, encoding="utf-8")
    print(proc.stdout.strip())


if __name__ == "__main__":
    main()
