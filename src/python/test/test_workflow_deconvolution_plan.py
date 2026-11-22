"""The plan's checks of design/validation/deconvolution.md (W-923): PL1 to PL8 with PL2b, the plan's part of SC5, SC8,
MT1 and EN1.

On ``input_root`` (``deconvolution_cases``: 2 FOVs × 2 rounds × 2 labels and the stain ``dapi`` in the reference round's
folder), planned with a ``deconvolution`` block. The plan reads no image and runs no task here; EN1 is Snakemake's dry
run of such a plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from starfinder.__main__ import main
from starfinder.workflow import plan
from starfinder.workflow._plan import fingerprint_components

from . import deconvolution_cases as D
from . import workflow_fixture as F
from . import workflow_variants as V

pytestmark = [pytest.mark.workflow, pytest.mark.validation]

FOVS = D.FOVS


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    work = tmp_path_factory.mktemp("deconvolution-plan")
    D.input_root(work)
    D.write_psf_files(work / "psf")
    return work


@pytest.fixture
def work(root, tmp_path):
    target = tmp_path / "work"
    shutil.copytree(root, target)
    return target


def planned(work, name, config):
    """The manifest of ``config`` (warnings recorded); (manifest, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest, _ = D.plan(work, name, config)
    return manifest, [str(w.message) for w in caught if "deconvolution" in str(w.message)]


def by_kind(manifest, kind):
    return [t for t in manifest["tasks"] if t["kind"] == kind]


def tasks(manifest):
    return {t["id"]: t for t in manifest["tasks"]}


def copy_path(manifest):
    return manifest["deconvolution"]["copy"]


# --- PL1: no block, no change -------------------------------------------------------------------------------------

def test_pl1_a_plan_without_the_block_has_no_stage(tmp_path):
    """Every worked-example variant without the block: no psf or deconvolve task, no ``deconvolved`` entry, no
    manifest key ``deconvolution`` and no execution key of the stage in the resolved configuration. The committed C1
    manifests and every task's hashes are compared by test_workflow_plan.py (``test_c1_*``), unchanged."""
    work = tmp_path / "tiny"
    F.golden().write_fixture(work)
    F.write_supplied(work)
    for variant, config in V.variants(work).items():
        if variant not in V.VARIANTS:
            continue
        manifest = plan(F.write_config(work, variant, config))
        assert "deconvolution" not in manifest
        assert not {"psf", "deconvolve"} & {t["kind"] for t in manifest["tasks"]}
        assert not any("deconvolved" in t or "deconvolution" in t for t in manifest["tasks"])
        resolved = json.loads((Path(manifest["roots"]["output"]) / "workflow" / "resolved.json").read_text())
        assert "deconvolution" not in resolved["configuration"]
        assert not {"deconwolf", "deconwolf_wisdom"} & set(resolved["configuration"]["execution"]), variant


# --- PL2, PL2b: the graph -------------------------------------------------------------------------------------------

SUPPLIED = {"steps": [{"method": "percentile_normalization", "p_low": 1.0, "p_high": 99.9, "fit": "supplied"}]}


def test_pl2_the_graph(work):
    config = D.configuration(work, "pl2", deconvolution=D.block(), morphology=True)
    config["processing"]["preprocessing"] = copy.deepcopy(SUPPLIED)
    manifest, _ = planned(work, "pl2", config)
    copy_ = copy_path(manifest)
    (psf,) = by_kind(manifest, "psf")
    assert psf["id"] == "psf" and psf["needs"] == []
    deconvolve = by_kind(manifest, "deconvolve")
    assert sorted(t["id"] for t in deconvolve) == sorted(f"deconvolve/{D.SAMPLE}/{fov}/{r}" for fov in FOVS
                                                         for r in D.ROUNDS)
    outputs = [o for t in manifest["tasks"] for o in t["outputs"]]
    assert len(outputs) == len(set(outputs))
    for task in deconvolve:
        fov, name = task["identity"]["fov"], task["identity"]["round"]
        assert task["needs"] == ["psf"] and task["group"] == f"deconvolve.{name}"
        records = f"{copy_}/records/{name}/{fov}/"
        assert sorted(o for o in task["outputs"] if o.startswith(records)) == sorted(
            [records + "deconvolution.json"] + [records + f"{label}.dw.log" for label in D.LABELS])
    for task in by_kind(manifest, "fov") + by_kind(manifest, "summarize"):
        fov = task["identity"]["fov"]
        assert {f"deconvolve/{D.SAMPLE}/{fov}/{r}" for r in D.ROUNDS} <= set(task["needs"])
        images = [i for i in task["inputs"] if i.endswith(".tif")]
        assert sorted(images) == sorted(f"{copy_}/{r}/{fov}/{label}.tif" for r in D.ROUNDS for label in D.LABELS)
        assert not any(i.startswith("input:round") for i in task["inputs"])
        assert task["deconvolved"] == {r: f"{copy_}/{r}/{fov}" for r in D.ROUNDS}
    for task in by_kind(manifest, "morphology"):
        fov = task["identity"]["fov"]
        assert f"input:round1/{fov}/{fov}_dapi.tif" in task["inputs"] and "deconvolved" not in task
        assert not any(n.startswith("deconvolve/") for n in task["needs"])
    # With reference_stain deconvolved too, the morphology task reads the copy's stain file and needs its task.
    config["deconvolution"]["rounds"] = [*D.ROUNDS, "reference_stain"]
    manifest, _ = planned(work, "pl2", config)
    copy_ = copy_path(manifest)
    for task in by_kind(manifest, "morphology"):
        fov = task["identity"]["fov"]
        assert f"{copy_}/round1/{fov}/dapi.tif" in task["inputs"]
        assert f"input:round1/{fov}/{fov}_dapi.tif" not in task["inputs"]
        assert f"deconvolve/{D.SAMPLE}/{fov}/reference_stain" in task["needs"]
        assert task["deconvolved"] == {"reference_stain": f"{copy_}/round1/{fov}"}


def test_pl2b_the_reference_round_and_its_stains(work):
    config = D.configuration(work, "pl2b", deconvolution=D.block(rounds=[*D.ROUNDS, "reference_stain"]),
                             morphology=True)
    manifest, _ = planned(work, "pl2b", config)
    copy_, all_tasks = copy_path(manifest), tasks(manifest)
    for fov in FOVS:
        reference = all_tasks[f"deconvolve/{D.SAMPLE}/{fov}/round1"]
        stain = all_tasks[f"deconvolve/{D.SAMPLE}/{fov}/reference_stain"]
        assert set(reference["outputs"]) == {f"{copy_}/round1/{fov}/ch00.tif", f"{copy_}/round1/{fov}/ch01.tif",
                                             *(f"{copy_}/records/round1/{fov}/{n}" for n in
                                               ("deconvolution.json", "ch00.dw.log", "ch01.dw.log"))}
        assert set(stain["outputs"]) == {f"{copy_}/round1/{fov}/dapi.tif",
                                         *(f"{copy_}/records/reference_stain/{fov}/{n}" for n in
                                           ("deconvolution.json", "dapi.dw.log"))}
        assert not set(reference["outputs"]) & set(stain["outputs"])
        assert reference["group"] != stain["group"]
    groups = {g["name"]: g for g in manifest["groups"]}
    assert groups["deconvolve.reference_stain"]["outputs"] != groups["deconvolve.round1"]["outputs"]


# --- PL3, SC5 (plan), MT1: refusals ---------------------------------------------------------------------------------

def refused(work, capsys, config, name="refused"):
    """The plan command's exit status and message for ``config``."""
    path = D.write_config(work, name, config)
    status = main(["workflow", "plan", "--config", str(path)])
    return status, capsys.readouterr().err


def variant(work, change, **options):
    config = D.configuration(work, "pl3", deconvolution=D.block(), **options)
    change(config)
    return config


def _set(path, value):
    def change(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        if value is _DELETE:
            node.pop(path[-1], None)
        else:
            node[path[-1]] = value
    return change


_DELETE = object()
DB = "deconvolution"
REFUSALS = {
    "matlab": (_set(["backend"], "matlab"), "deconvolution"),
    "iterations absent": (_set([DB, "iterations"], _DELETE), "iterations"),
    "iterations 0": (_set([DB, "iterations"], 0), "deconvolution.iterations"),
    "iterations 2.5": (_set([DB, "iterations"], 2.5), "deconvolution.iterations"),
    "unknown round": (_set([DB, "rounds"], ["round9"]), "deconvolution.rounds"),
    "unread round": (_set([DB, "rounds"], ["reference_stain"]), "deconvolution.rounds"),
    "no voxel_size_xy": (_set(["voxel_size_xy"], _DELETE), "voxel_size_xy"),
    "no voxel_size_z": (_set(["voxel_size_z"], _DELETE), "voxel_size_z"),
    "no wavelengths": (_set([DB, "psf", "channels", "ch01"], _DELETE), "deconvolution.psf.channels"),
    "tile below the PSF": (_set([DB, "tile_size"], 48), "deconvolution.tile_size"),
    "dtype float32": (_set([DB, "dtype"], "float32"), "deconvolution.dtype"),
    "relative copy_root": (_set([DB, "copy_root"], "copies"), "deconvolution.copy_root"),
    "scale with uint8": (_set([DB, "scale"], 16), "deconvolution.scale"),
    "scale 0": (lambda c: c[DB].update(dtype="uint16", scale=0), "deconvolution.scale"),
    "scale mapping": (lambda c: c[DB].update(dtype="uint16", scale={"round1": {"ch00": 64, "ch01": 64},
                                                                    "round2": {"ch00": 64}}),
                      "deconvolution.scale"),
}


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_pl3_refusals(work, capsys, case):
    change, key = REFUSALS[case]
    status, message = refused(work, capsys, variant(work, change))
    assert status == 2 and "plan refused" in message and key in message, message


def file_config(work, **declared):
    files = {label: str(work / "psf" / f"psf_{label}.tif") for label in (*D.LABELS, D.STAIN)}
    block = D.file_block(files)
    block["psf"]["voxel_size_um"].update(declared)
    return D.configuration(work, "pl3", deconvolution=block)


@pytest.mark.contract
def test_pl3_refusals_of_supplied_files(work, capsys):
    for change, key in [
            (lambda c: c[DB]["psf"]["files"].pop("ch01"), "deconvolution.psf.files"),
            (lambda c: c[DB]["psf"]["files"].update(ch01=str(work / "absent.tif")), "deconvolution.psf.files"),
            (lambda c: c[DB]["psf"].update(voxel_size_um=None), "deconvolution.psf.voxel_size_um"),
            (lambda c: c.pop("voxel_size_xy"), "voxel_size_xy"),
            (lambda c: c.pop("voxel_size_z"), "voxel_size_z")]:
        config = file_config(work)
        change(config)
        status, message = refused(work, capsys, config)
        assert status == 2 and key in message, message
    # A declared spacing that differs by 2 parts in 10^6 is refused, in Y and X and in Z; 1 part in 10^7 is accepted.
    for key, value in (("xy", D.VOXEL_XY), ("z", D.VOXEL_Z)):
        status, message = refused(work, capsys, file_config(work, **{key: value * (1 + 2e-6)}))
        assert status == 2 and f"deconvolution.psf.voxel_size_um.{key}" in message, message
        status, message = refused(work, capsys, file_config(work, **{key: value * (1 + 1e-7)}))
        assert status == 0, message
    # The tile size at the PSF's lateral size plus deconwolf's padding is accepted.
    status, message = refused(work, capsys, variant(work, _set([DB, "tile_size"], 49)))
    assert status == 0, message


@pytest.mark.contract
@pytest.mark.parametrize("stain", ["ch00", "ch0"])
def test_pl3_labels_confused_in_one_image_folder(work, capsys, stain):
    """A stain label equal to a sequencing label of the reference round, or one whose pattern matches it, with both
    rounds deconvolved. An equal label is refused before the stage by the translation's own rule (a reference stain is
    not a sequencing colour), naming reference_stains; a label whose pattern matches another's file by the stage,
    naming deconvolution.rounds and both labels."""
    config = variant(work, _set([DB, "rounds"], [*D.ROUNDS, "reference_stain"]), morphology=True)
    config["reference_stains"] = [{"channel": stain, "name": "DAPI"}]
    config[DB]["psf"]["channels"][stain] = {"excitation_nm": 405, "emission_nm": 461}
    status, message = refused(work, capsys, config)
    key = "reference_stains share the patterns ['ch00']" if stain == "ch00" else "deconvolution.rounds"
    assert status == 2 and key in message and repr(stain) in message, message


@pytest.mark.contract
def test_mt1_matlab_is_untouched(work, capsys):
    """backend: matlab refuses the block (PL3); the MATLAB rule set and its schema (workflow/, but the Python rules of
    rules/tasks.smk) are byte-identical to v0.3.0 (70edc72)."""
    status, message = refused(work, capsys, variant(work, _set(["backend"], "matlab")))
    assert status == 2 and "backend: matlab refuses the deconvolution block" in message
    git = shutil.which("git")
    if git is None:
        pytest.skip("the comparison with v0.3.0 (70edc72) needs Git on PATH")
    base = "70edc72f5e8ecf16f48250159a8a22096383c846"
    if subprocess.run([git, "-C", str(F.ROOT), "cat-file", "-e", f"{base}^{{commit}}"]).returncode != 0:
        pytest.skip("the comparison with v0.3.0 needs the revision 70edc72, which this clone does not hold")
    listed = subprocess.run([git, "-C", str(F.ROOT), "ls-tree", "-r", base, "--", "workflow"], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    files = {line.split("\t")[1]: line.split()[2] for line in listed}
    matlab = sorted(p for p in files if p != "workflow/rules/tasks.smk")
    assert "workflow/rules/legacy.smk" in matlab and "workflow/schemas/config.schema.yaml" in matlab
    for path in matlab:
        data = (F.ROOT / path).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == files[path], path


# --- PL4, PL5, PL8: what is hashed ----------------------------------------------------------------------------------

def hashes(manifest):
    """(copy id, {task id: (blocks_sha256, fingerprint)}) of a plan."""
    return manifest["deconvolution"]["copy_id"], {t["id"]: (t["blocks_sha256"], t["fingerprint"])
                                                    for t in manifest["tasks"]}


def consumer_ids(manifest):
    return [t["id"] for t in manifest["tasks"] if t.get("deconvolved")]


CHANGES = {
    "iterations": lambda c: c[DB].update(iterations=4),
    "tile_size": lambda c: c[DB].update(tile_size=64),
    "dtype": lambda c: c[DB].update(dtype="uint16"),
    "border_quality": lambda c: c[DB].update(border_quality=1),
    "rounds": lambda c: c[DB].update(rounds=list(D.ROUNDS)),
    "psf parameter": lambda c: c[DB]["psf"].update(na=1.25),
    "voxel_size_xy": lambda c: c.update(voxel_size_xy=0.11),
    "voxel_size_z": lambda c: c.update(voxel_size_z=0.31),
}
NO_CHANGE = {
    "threads": lambda c: c["execution"].update(tasks={"fov": {"threads": 1}, "deconvolve": {"threads": 2}}),
    "gpu": lambda c: c["execution"].update(tasks={"fov": {"threads": 1}, "deconvolve": {"gpu": True}}),
    "deconwolf": lambda c: c["execution"].update(deconwolf="/opt/other/dw"),
    "deconwolf_wisdom": lambda c: c["execution"].update(deconwolf_wisdom="/opt/wisdom"),
}


def test_pl4_what_is_hashed(work):
    base = D.configuration(work, "pl4", deconvolution=D.block())
    before = hashes(planned(work, "pl4", base)[0])
    for name, change in CHANGES.items():
        config = copy.deepcopy(base)
        change(config)
        manifest = planned(work, "pl4", config)[0]
        after = hashes(manifest)
        assert after[0] != before[0], name
        for task in manifest["tasks"]:
            if task["kind"] in ("psf", "deconvolve") or task["id"] in consumer_ids(manifest):
                assert after[1][task["id"]][1] != before[1][task["id"]][1], (name, task["id"])
    for name, change in NO_CHANGE.items():
        config = copy.deepcopy(base)
        change(config)
        assert hashes(planned(work, "pl4", config)[0]) == before, name
    # Under dtype uint16, scale null and scale 64 give one copy id; another scale another.
    uint16 = {scale: hashes(planned(work, "pl4", dict(base, deconvolution=dict(base[DB], dtype="uint16",
                                                                                 scale=scale)))[0])[0]
              for scale in (None, 64, 32)}
    assert uint16[None] == uint16[64] != uint16[32]


def test_pl4_a_rewritten_psf_file_changes_the_copy_id(work):
    config = file_config(work)
    first = hashes(planned(work, "pl4-file", config)[0])[0]
    D.write_psf_files(work / "psf", array=D.psf_small((1, 0, 0)))
    assert hashes(planned(work, "pl4-file", config)[0])[0] != first


def test_pl5_selection_growth_keeps_the_first_fovs_tasks(work):
    config = D.configuration(work, "pl5", deconvolution=D.block(), selected_fovs=[FOVS[0]])
    one = hashes(planned(work, "pl5", config)[0])[1]
    config["selected_fovs"] = None
    both = hashes(planned(work, "pl5", config)[0])[1]
    kept = [i for i in one if i == "psf" or i.startswith(f"deconvolve/{D.SAMPLE}/{FOVS[0]}/")]
    assert len(kept) == 3 and all(one[i] == both[i] for i in kept)


def expected_id(manifest, resolved):
    """The copy id from the specification's seven fields, computed here: the first 12 hex digits of the SHA-256 of
    their canonical JSON (keys sorted, no whitespace, UTF-8)."""
    r = resolved["configuration"]
    fields = {"dataset_id": r["dataset_id"], "sample_id": r["sample_id"],
              "image_dir_template": r["image_dir_template"], "voxel_size_xy": r["voxel_size_xy"],
              "voxel_size_z": r["voxel_size_z"],
              "deconvolution": {k: v for k, v in r["deconvolution"].items() if k != "copy_root"}, "psf_files": None}
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@pytest.mark.parametrize("place", ["output", "beside", "absolute"])
def test_pl6_the_copys_place(work, place):
    root = work / "copies" if place == "absolute" else place
    config = D.configuration(work, "pl6", deconvolution=D.block(copy_root=str(root)))
    manifest = planned(work, "pl6", config)[0]
    out = Path(manifest["roots"]["output"])
    resolved = json.loads((out / "workflow" / "resolved.json").read_text())
    identifier = expected_id(manifest, resolved)
    assert manifest["deconvolution"]["copy_id"] == identifier
    expected = {"output": f"deconvolved/{D.SAMPLE}/dw-{identifier}",
                "beside": str(work / "input" / f"{D.DATASET_ID}-dw" / D.SAMPLE / f"dw-{identifier}"),
                "absolute": str(work / "copies" / D.DATASET_ID / D.SAMPLE / f"dw-{identifier}")}[place]
    assert copy_path(manifest) == expected
    for task in by_kind(manifest, "deconvolve"):
        assert all(o.startswith(expected + "/") for o in task["outputs"])
        assert all(Path(o).is_absolute() == (place != "output") for o in task["outputs"])


def test_pl7_a_saved_set_needs_no_sequencing_copy(work):
    saved = work / "saved"
    for fov in FOVS:
        folder = saved / "fovs" / fov
        (folder / "checkpoints" / "registered").mkdir(parents=True)
        (folder / "run.json").write_text("{}")
        (folder / "checkpoints" / "registered" / "transforms.json").write_text("{}")
    config = D.configuration(work, "pl7", deconvolution=D.block())
    config["processing"]["spot_finding_input"] = str(saved)
    manifest = planned(work, "pl7", config)[0]
    assert by_kind(manifest, "deconvolve") == [] and len(by_kind(manifest, "fov")) == 2
    assert manifest["deconvolution"]["sequencing"] is True


def test_pl8_the_fingerprints_inputs(work):
    """With an absolute copy_root and no copy yet, the plan succeeds; the consumers' fingerprints stat none of the
    copy's files and change through their needs when a deconvolve task's fingerprint changes."""
    config = D.configuration(work, "pl8", deconvolution=D.block(copy_root=str(work / "copies")))
    manifest = planned(work, "pl8", config)[0]
    copy_ = copy_path(manifest)
    assert not Path(copy_).exists()
    fingerprints = {}
    for task in manifest["tasks"]:
        components = fingerprint_components(task, manifest, fingerprints)
        fingerprints[task["id"]] = task["fingerprint"]
        if task.get("deconvolved"):
            assert components["inputs"] == [] and any(i.startswith(copy_) for i in task["inputs"])
    before = hashes(manifest)[1]
    raw = work / "input" / D.DATASET_ID / D.SAMPLE / "round2" / FOVS[0]
    path = raw / f"{FOVS[0]}_ch00.tif"
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    after = hashes(planned(work, "pl8", config)[0])[1]
    changed = {i for i in before if before[i][1] != after[i][1]}
    assert changed == {f"deconvolve/{D.SAMPLE}/{FOVS[0]}/round2", f"fov/{D.SAMPLE}/{FOVS[0]}"}


# --- SC8: the warning for intensities across FOVs -----------------------------------------------------------------

FEEDS = {
    "supplied": lambda c: c["processing"].update(preprocessing=copy.deepcopy(SUPPLIED)),
    "global": lambda c: c["processing"]["spot_finding"].update(threshold_mode="global", threshold_value=0.2),
    "absolute": lambda c: c["processing"]["spot_finding"].update(threshold_mode="absolute", threshold_value=50.0),
}


@pytest.mark.parametrize("feed", sorted(FEEDS))
def test_sc8_the_warning_for_intensities_across_fovs(work, feed):
    config = D.configuration(work, "sc8", deconvolution=D.block())
    FEEDS[feed](config)
    manifest, found = planned(work, "sc8", config)
    assert len(found) == 1 and "dtype: uint16" in found[0] and by_kind(manifest, "deconvolve")
    config[DB].update(dtype="uint16")
    assert planned(work, "sc8", config)[1] == []


def test_sc8_no_warning_without_a_feed(work):
    assert planned(work, "sc8", D.configuration(work, "sc8", deconvolution=D.block()))[1] == []


# --- EN1: Snakemake ----------------------------------------------------------------------------------------------

def test_en1_snakemake(work):
    from .workflow_snakemake import snakemake
    config = D.configuration(work, "en1", deconvolution=D.block(rounds=[*D.ROUNDS, "reference_stain"]),
                             morphology=True)
    config["execution"]["tasks"]["deconvolve"] = {"gpu": True}
    path = D.write_config(work, "en1", config)
    done = snakemake(work, path, "-n")
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads((Path(work) / "output" / D.DATASET_ID / "en1" / "workflow" / "manifest.json").read_text())
    groups = [g["name"] for g in manifest["groups"]]
    # One block per job, in Snakemake's order: its rule, wildcards, resources and the command's task.
    rules, order = {}, []
    for block in re.split(r"\n(?=(?:local)?rule \w+:)", done.stdout):
        found = re.search(r"--task (\S+)", block)
        if not found:
            continue
        task = found.group(1)
        wildcards = re.search(r"^    wildcards: (.*)$", block, re.M)
        resources = re.search(r"^    resources: (.*)$", block, re.M)
        rules[task] = (re.search(r"(?:^|\n)(?:local)?rule (\w+):", block).group(1),
                       dict(re.findall(r"(\w+)=([^,\s]+)", wildcards.group(1))) if wildcards else {},
                       resources.group(1) if resources else "")
        order.append(task)
    assert sorted(rules) == sorted(t["id"] for t in manifest["tasks"])
    for task in by_kind(manifest, "deconvolve"):
        rule, wildcards, resources = rules[task["id"]]
        index = groups.index(task["group"])
        assert rule.startswith(f"deconvolve_{index}")
        expected = {"fov"} | ({"round"} if task["identity"]["round"] != "reference_stain" else set())
        assert set(wildcards) == expected, (task["id"], wildcards)
        assert "gpu=1" in resources
        assert order.index("psf") < order.index(task["id"])
    for task in manifest["tasks"]:
        for need in task["needs"]:
            assert order.index(need) < order.index(task["id"])
