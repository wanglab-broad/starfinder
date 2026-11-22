"""The engine checks of design/validation/deconvolution.md (W-923): EN2 (Nextflow's stub run) and EN3 (Snakemake and
Nextflow write byte-equal copies, records and consumer outputs).

On ``input_root`` (``deconvolution_cases``) with a ``deconvolution`` block, the stand-in ``dw`` and bounded resources.
Both need Nextflow 26.10.4 or newer and Java 17 or newer on PATH and skip without them; EN3 also needs Snakemake.
"""
from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from . import deconvolution_cases as D
from . import workflow_nextflow as N
from .workflow_snakemake import TRIGGERS, snakemake

pytestmark = [pytest.mark.workflow, pytest.mark.validation, pytest.mark.nextflow, pytest.mark.slow]

SUPPLIED = {"steps": [{"method": "percentile_normalization", "p_low": 1.0, "p_high": 99.9, "fit": "supplied"}]}
#: The keys of a round record that say where and when it was made, or how long it took (left out of EN3's comparison):
#: the arguments name the task's temporary folder.
VARYING = ("started_at", "ended_at", "host", "job_id")
VARYING_CHANNEL = ("wall_s", "cpu_s", "arguments")


@pytest.fixture
def work(tmp_path):
    N.executable()
    D.input_root(tmp_path)
    D.write_psf_files(tmp_path / "psf")
    return tmp_path


def configuration(work, output_id, *, gpu=False):
    dw, _ = D.stand_in(work / "bin")
    files = {label: str(work / "psf" / f"psf_{label}.tif") for label in (*D.LABELS, D.STAIN, "ch02")}
    config = D.configuration(work, output_id, deconvolution=D.file_block(files, rounds=[*D.ROUNDS, "reference_stain"]),
                             dw=dw, morphology=True)
    config["processing"]["preprocessing"] = copy.deepcopy(SUPPLIED)
    config = N.small_resources(config)
    config["execution"]["deconwolf"] = str(dw)
    if gpu:
        config["execution"]["tasks"]["deconvolve"]["gpu"] = True
    return config


@pytest.mark.parametrize("gpu", [False, True])
def test_en2_nextflow(work, gpu):
    config = D.write_config(work, "en2", configuration(work, "en2", gpu=gpu))
    done, rows = N.run(work, config)
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = N.manifest(config)
    rows = N.tasks(rows)
    assert Counter(r["tag"] for r in rows) == Counter(t["id"] for t in manifest["tasks"])
    by_id = {r["tag"]: r for r in rows}
    names = {r["tag"]: r["name"].split(" ")[0] for r in rows}
    assert names["psf"] == "PSF"
    deconvolve = [t["id"] for t in manifest["tasks"] if t["kind"] == "deconvolve"]
    assert {names[t] for t in deconvolve} == {"DECONVOLVE_GPU" if gpu else "DECONVOLVE"}
    summarize = [t["id"] for t in manifest["tasks"] if t["kind"] == "summarize"]
    assert summarize and all(names[t] == "SUMMARIZE_0" for t in summarize)
    for task in manifest["tasks"]:
        for need in task["needs"]:
            assert int(by_id[task["id"]]["start"]) >= int(by_id[need]["complete"]), (task["id"], need)
    assert all(int(by_id[t]["start"]) >= int(by_id["psf"]["complete"]) for t in deconvolve)
    first = min(int(by_id[t]["start"]) for t in summarize)
    fovs = {t["identity"]["fov"] for t in manifest["tasks"] if t["kind"] == "summarize"}
    assert all(int(by_id[t]["complete"]) <= first for t in deconvolve
               if manifest_task(manifest, t)["identity"]["fov"] in fovs and
               manifest_task(manifest, t)["identity"]["round"] != "reference_stain")


def manifest_task(manifest, task_id):
    return next(t for t in manifest["tasks"] if t["id"] == task_id)


def snapshot(manifest):
    """The copy's files (records less their varying fields) and the consumers' declared outputs but run.json."""
    out = Path(manifest["roots"]["output"])
    copy_ = D.copy_folder(manifest)
    files = {}
    for path in sorted(copy_.rglob("*")):
        if not path.is_file():
            continue
        name = "copy/" + path.relative_to(copy_).as_posix()
        if path.name == "deconvolution.json":
            record = json.loads(path.read_text())
            for key in VARYING:
                record.pop(key)
            for channel in record["channels"]:
                for key in VARYING_CHANNEL:
                    channel.pop(key)
            files[name] = record
        else:
            files[name] = path.read_bytes()
    for task in manifest["tasks"]:
        if task.get("deconvolved"):
            for output in task["outputs"]:
                path = Path(output) if Path(output).is_absolute() else out / output
                if path.name not in ("run.json", "nuclei_registration.json") and path.is_file():
                    files[output] = path.read_bytes()
    return files


def test_en3_the_engines_agree(work):
    config = D.write_config(work, "en3", configuration(work, "en3"))
    done = snakemake(work, config, "--rerun-incomplete", *TRIGGERS)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    manifest = N.manifest(config)
    first = snapshot(manifest)
    assert any(name.startswith("copy/round1/") for name in first)
    shutil.rmtree(manifest["roots"]["output"])
    done, rows = N.run(work, config, stub=False)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    second = snapshot(N.manifest(config))
    assert sorted(first) == sorted(second)
    assert [name for name in first if first[name] != second[name]] == []
