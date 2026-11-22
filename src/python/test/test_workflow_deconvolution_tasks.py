"""The task checks of design/validation/deconvolution.md (W-923): SC1 to SC7 with SC1b, PS1 to PS4, WI1 to WI7, RC1 to
RC5, RU1 to RU11 and DW1 to DW3.

The psf and deconvolve tasks run in this process on ``input_root`` (``deconvolution_cases``) with the stand-in ``dw``,
which writes ``y = x + 5`` untiled and ``y = x`` tiled, so that every expected copy is an exact integer array.
PS3, PS4 and RU11 need psfmodels (the ``deconvolution`` extra) and skip without it; DW1 to DW3 need the deconwolf
binary (``dw`` on PATH or ``STARFINDER_TEST_DECONWOLF``) and skip without it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.io import load_round, load_volume
from starfinder.io.tiff import ImageLoadConfig
from starfinder.workflow._deconvolve import (centre_psf, cpu_identity, deconwolf_identity, fwhm, wisdom_fields,
    wisdom_key, wisdom_name)

from . import deconvolution_cases as D
from . import workflow_fixture as F

pytestmark = [pytest.mark.workflow, pytest.mark.validation]

FOV = D.FOVS[0]
SAMPLE = D.SAMPLE


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """``input_root`` and ``input_root_stored``, written once, and the PSF files."""
    out = {}
    for stored in (False, True):
        work = tmp_path_factory.mktemp("stored" if stored else "plain")
        D.input_root(work, stored=stored)
        D.write_psf_files(work / "psf")
        out[stored] = work
    return out


@pytest.fixture
def work(roots, tmp_path, monkeypatch):
    target = tmp_path / "work"
    shutil.copytree(roots[False], target)
    F.fix_code(monkeypatch)
    return target


@pytest.fixture
def stored_work(roots, tmp_path, monkeypatch):
    target = tmp_path / "stored"
    shutil.copytree(roots[True], target)
    F.fix_code(monkeypatch)
    return target


def psf_files(work):
    return {label: str(work / "psf" / f"psf_{label}.tif") for label in (*D.LABELS, D.STAIN, "ch02")}


def block(work, **keys):
    """A deconvolution block with the supplied PSF files of the fixture (psf_small) and ``keys``."""
    return D.file_block(psf_files(work), **keys)


def stage(work, name, deconvolution, dw, *, kinds=("psf", "deconvolve"), selected=(FOV,), **options):
    """Plan ``deconvolution`` on one FOV and run the tasks of ``kinds``; (manifest, manifest path, copy, statuses)."""
    config = D.configuration(work, name, deconvolution=deconvolution, dw=dw, selected_fovs=list(selected), **options)
    manifest, path = D.plan(work, name, config)
    statuses = D.run_kinds(manifest, path, *kinds)
    return manifest, path, D.copy_folder(manifest), statuses


def raw(work, name, label, fov=FOV):
    folder = "round1" if name == "reference_stain" else name
    return work / "input" / D.DATASET_ID / SAMPLE / folder / fov / f"{fov}_{label}.tif"


def record(copy, name, fov=FOV):
    return json.loads((copy / "records" / name / fov / "deconvolution.json").read_text())


def task_record(path, task_id):
    return json.loads((path.parents[1] / "workflow" / "tasks" / f"{task_id.replace('/', '.')}.json").read_text())


def failed_record(path, task_id):
    return json.loads((path.parents[1] / "workflow" / "tasks" / f"{task_id.replace('/', '.')}.failed.json").read_text())


def deconvolve_id(name, fov=FOV):
    return f"deconvolve/{SAMPLE}/{fov}/{name}"


def stored_values(path):
    return tifffile.imread(path)


def expected_uint8(arrays):
    """SC1: one factor f = 255 / M for the round, M the maximum over its channels; rint(x × f) clipped to 0–255."""
    m = max(float(a.max()) for a in arrays.values())
    f = 255.0 / m if m > 0 else 1.0
    return {k: np.clip(np.rint(a.astype(np.float64) * f), 0, 255).astype(np.uint8) for k, a in arrays.items()}, f, m


def expected_uint16(arrays, scales):
    return {k: np.clip(np.rint(a.astype(np.float64) * scales[k]), 0, 65535).astype(np.uint16)
            for k, a in arrays.items()}


def raw_arrays(work, name="round1", labels=D.LABELS):
    return {label: tifffile.imread(raw(work, name, label)) for label in labels}


# --- SC1, SC1b, SC2: the offset, the factor and the scale -----------------------------------------------------------

@pytest.mark.parametrize("tiled", [False, True])
def test_sc1_sc2_offset_and_stretch_factor(work, tiled):
    dw, _ = D.stand_in(work / "bin")
    keys = {"rounds": ["round1"], **({"tile_size": 64} if tiled else {})}
    manifest, path, copy, statuses = stage(work, "sc1", block(work, **keys), dw)
    assert set(statuses.values()) == {0}
    expected, f, m = expected_uint8(raw_arrays(work))
    rec = record(copy, "round1")
    assert rec["storage"]["dtype"] == "uint8" and rec["storage"]["round_max"] == m
    assert rec["storage"]["stretch_factor"] == pytest.approx(f, rel=1e-12, abs=0)
    for channel in rec["channels"]:
        assert channel["kept_offset"] == (0 if tiled else 5)
        stored = stored_values(copy / "round1" / FOV / f"{channel['label']}.tif")
        assert stored.dtype == np.uint8 and np.array_equal(stored, expected[channel["label"]])


def test_sc1_a_round_of_zeros_stores_zeros(work):
    for label in D.LABELS:
        tifffile.imwrite(raw(work, "round2", label), np.zeros(D.SHAPE, dtype=np.uint8))
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "sc1-zero", block(work, rounds=["round2"]), dw)
    rec = record(copy, "round2")
    assert rec["storage"] == {"dtype": "uint8", "stretch_factor": 1.0, "round_max": 0.0}
    for label in D.LABELS:
        assert not stored_values(copy / "round2" / FOV / f"{label}.tif").any()


@pytest.mark.parametrize("tiled", [False, True])
def test_sc1b_sc2_offset_and_scale_uint16(work, tiled):
    dw, _ = D.stand_in(work / "bin")
    keys = {"rounds": ["round1"], "dtype": "uint16", "scale": 64, **({"tile_size": 64} if tiled else {})}
    manifest, path, copy, statuses = stage(work, "sc1b", block(work, **keys), dw)
    expected = expected_uint16(raw_arrays(work), {label: 64 for label in D.LABELS})
    rec = record(copy, "round1")
    assert rec["storage"] == {"dtype": "uint16"}
    for channel in rec["channels"]:
        assert channel["kept_offset"] == (0 if tiled else 5) and channel["scale"] == 64
        stored = stored_values(copy / "round1" / FOV / f"{channel['label']}.tif")
        assert stored.dtype == np.uint16 and np.array_equal(stored, expected[channel["label"]])


# --- SC3: the offset rule is checked --------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["minimum 0", "version 0.4.6"])
def test_sc3_the_offset_rule_is_checked(work, case):
    options = {"min_zero": True} if case == "minimum 0" else {"version": "0.4.6"}
    dw, _ = D.stand_in(work / "bin", **options)
    manifest, path, copy, statuses = stage(work, "sc3", block(work, rounds=["round1"]), dw)
    task = deconvolve_id("round1")
    assert statuses["psf"] == 0 and statuses[task] == 1
    failed = failed_record(path, task)
    assert ("kept offset rule" if case == "minimum 0" else "'0.4.6'") in failed["error"]["message"]
    assert not list((copy / "round1").rglob("*.tif"))
    assert not (copy / "records" / "round1" / FOV / "deconvolution.json").exists()
    assert not (path.parents[1] / "workflow" / "tasks" / f"{task.replace('/', '.')}.json").exists()
    declared = {o for t in manifest["tasks"] if t["id"] == task for o in t["outputs"]}
    assert set(failed["removed"]) <= declared


def test_sc3_a_tiled_minimum_of_5_does_not_fail(work):
    dw, _ = D.stand_in(work / "bin", shift=5)
    manifest, path, copy, statuses = stage(work, "sc3-tiled", block(work, rounds=["round1"], tile_size=64), dw)
    assert statuses[deconvolve_id("round1")] == 0
    arrays = raw_arrays(work)
    for channel in record(copy, "round1")["channels"]:
        assert channel["float_min"] == float(arrays[channel["label"]].min()) + 5


# --- SC4, SC5: saturation and a scale per round and label -----------------------------------------------------------

def test_sc4_saturation_and_negatives(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "sc4", block(work, rounds=["round1"], dtype="uint16", scale=1000),
                                           dw)
    arrays = raw_arrays(work)
    rec = record(copy, "round1")
    for channel in rec["channels"]:
        x = arrays[channel["label"]].astype(np.float64)
        stored = stored_values(copy / "round1" / FOV / f"{channel['label']}.tif")
        assert channel["clipped_high"] == int((np.rint(x * 1000) > 65535).sum()) > 0
        assert np.array_equal(stored[np.rint(x * 1000) > 65535], np.full(channel["clipped_high"], 65535))
    log = (path.parents[1] / "workflow" / "logs" / f"{deconvolve_id('round1').replace('/', '.')}.log").read_text()
    assert log.count("saturated at 65535") == 1
    # A stand-in output below the offset (tiled, y = x - 3): the negative values store 0 and are counted.
    for dtype in ("uint16", "uint8"):
        below, _ = D.stand_in(work / f"bin-{dtype}", shift=-3)
        keys = {"rounds": ["round1"], "tile_size": 64, "dtype": dtype, **({"scale": 64} if dtype == "uint16" else {})}
        manifest, path, copy, statuses = stage(work, f"sc4-{dtype}", block(work, **keys), below)
        for channel in record(copy, "round1")["channels"]:
            x = arrays[channel["label"]].astype(np.float64) - 3
            stored = stored_values(copy / "round1" / FOV / f"{channel['label']}.tif")
            assert channel["clipped_low"] == int((x < 0).sum()) > 0
            assert not stored[x <= 0].any()


def test_sc5_a_scale_per_round_and_label(work):
    dw, _ = D.stand_in(work / "bin")
    scales = {"ch00": 10, "ch01": 20}
    manifest, path, copy, statuses = stage(
        work, "sc5", block(work, rounds=["round1"], dtype="uint16", scale={"round1": scales}), dw)
    expected = expected_uint16(raw_arrays(work), scales)
    for channel in record(copy, "round1")["channels"]:
        assert channel["scale"] == scales[channel["label"]]
        assert np.array_equal(stored_values(copy / "round1" / FOV / f"{channel['label']}.tif"),
                              expected[channel["label"]])


# --- SC6, SC7: the file, its metadata, and the copy loads as the raw round loads ----------------------------------

def tags(path):
    with tifffile.TiffFile(path) as tif:
        return {int(page.compression) for page in tif.pages}, any(317 in page.tags for page in tif.pages), \
            tif.series[0].axes, "starfinder_metadata" in json.loads(tif.pages[0].description or "{}")


@pytest.mark.parametrize("fixture", ["plain", "stored"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_sc6_the_file_and_its_metadata(work, stored_work, fixture, dtype):
    work = stored_work if fixture == "stored" else work
    dw, _ = D.stand_in(work / "bin")
    keys = {"rounds": ["round1", "reference_stain"], "dtype": dtype}
    manifest, path, copy, statuses = stage(work, "sc6", block(work, **keys), dw, morphology=True)
    folder = copy / "round1" / FOV
    assert sorted(p.name for p in folder.iterdir()) == ["ch00.tif", "ch01.tif", "dapi.tif"]
    for label in (*D.LABELS, D.STAIN):
        codes, predictor, axes, has_metadata = tags(folder / f"{label}.tif")
        assert codes == {8} and not predictor and axes == "ZYX"
        assert has_metadata is (fixture == "stored")
        copied = load_volume(folder / f"{label}.tif")
        source = load_volume(raw(work, "round1", label))
        assert copied.image.dtype == np.dtype(dtype) and copied.image.shape == D.SHAPE
        assert copied.diagnostics["metadata_source"] == source.diagnostics["metadata_source"] == (
            "stored" if fixture == "stored" else "unknown")
        if fixture == "stored":
            assert copied.metadata == source.metadata


def test_sc6_the_copy_follows_image_compression(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "sc6-none", block(work, rounds=["round1"]), dw,
                                           execution={"image_compression": "none"})
    rec = record(copy, "round1")
    assert rec["image_compression"] == "none"
    assert all(tags(copy / "round1" / FOV / f"{label}.tif")[0] == {1} for label in D.LABELS)


@pytest.mark.parametrize("fixture", ["plain", "stored"])
def test_sc7_the_copy_loads_as_the_raw_round_loads(work, stored_work, fixture):
    work = stored_work if fixture == "stored" else work
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "sc7", block(work, rounds=[*D.ROUNDS, "reference_stain"]), dw,
                                           kinds=("psf", "deconvolve", "fov"), morphology=True)
    assert set(statuses.values()) == {0}, statuses
    raw_root = work / "input" / D.DATASET_ID / SAMPLE
    for name, labels in (("round1", (*D.LABELS, D.STAIN)), ("round2", D.LABELS)):
        config = ImageLoadConfig(channel_labels=labels)
        source = load_round(raw_root / name / FOV, config=config)
        copied = load_round(copy / name / FOV, config=config)
        assert copied.channel_labels == labels and copied.image.dtype == np.uint8
        for key in ("spacing_zyx", "origin_zyx", "direction_zyx", "spatial_unit"):
            assert getattr(copied.metadata, key) == getattr(source.metadata, key)
        if fixture == "plain":
            assert copied.metadata.frame_id == str((copy / name / FOV).resolve())
            assert source.metadata.frame_id == str((raw_root / name / FOV).resolve())
        else:
            assert copied.metadata == source.metadata
        expected, _, _ = expected_uint8(raw_arrays(work, name, D.LABELS))
        for c, label in enumerate(D.LABELS):
            assert np.array_equal(copied.image[..., c], expected[label])
    if fixture == "plain":
        # The negative control: the raw files' path-built metadata stored in each copy file of a plain round.
        from starfinder.io import save_volume
        control = work / "control"
        for label in D.LABELS:
            loaded = load_volume(raw(work, "round2", label))
            save_volume(tifffile.imread(copy / "round2" / FOV / f"{label}.tif"), control / f"{label}.tif",
                        metadata=loaded.metadata)
        with pytest.raises(ValueError, match="channel frames differ"):
            load_round(control, config=ImageLoadConfig(channel_labels=D.LABELS))


# --- PS1, PS2: the PSF's centring and a supplied file ---------------------------------------------------------------

@pytest.mark.contract
def test_ps1_centring():
    # (a) zero support, rolled: back to the centre losing nothing.
    centred, rec = centre_psf(D.psf_core((2, 1, 0)))
    assert rec["shift_zyx"] == [-2, -1, 0] and rec["peak_after"] == [3, 4, 4] and rec["centre"] == [3, 4, 4]
    assert rec["peak_offset_before"] == [2, 1, 0]
    assert rec["sum_dropped"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(centred.astype(np.float32), D.psf_core(), rtol=1e-6, atol=1e-6)
    # (b) full support, rolled: the slices the roll wraps (result planes 5-6, row 8) are zeroed.
    centred, rec = centre_psf(D.psf_small((2, 1, 0)))
    t = D.psf_small().astype(np.float64)
    t[5:7] = 0
    t[:, 8, :] = 0
    expected = t / t.sum()
    assert rec["shift_zyx"] == [-2, -1, 0] and rec["peak_after"] == [3, 4, 4]
    np.testing.assert_allclose(centred.astype(np.float32), expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert rec["sum_dropped"] == pytest.approx(1 - t.sum(), rel=1e-6, abs=1e-6)
    assert float(centred.sum()) == pytest.approx(1.0, abs=1e-6)
    # (c) unrolled: nothing moves and nothing is zeroed.
    for fixture in (D.psf_small(), D.psf_core()):
        centred, rec = centre_psf(fixture)
        assert rec["shift_zyx"] == [0, 0, 0] and rec["sum_dropped"] == 0.0
        np.testing.assert_allclose(centred.astype(np.float32), fixture, rtol=1e-6, atol=1e-6)


def test_ps2_a_supplied_file(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "ps2", block(work, rounds=["round1"]), dw, kinds=("psf",))
    entry = json.loads((copy / "psf" / "psf.json").read_text())["labels"]["ch00"]
    source = psf_files(work)["ch00"]
    assert entry["path"] == source and entry["file_sha256"] == D.sha256(source)
    assert entry["sha256"] == hashlib.sha256(np.ascontiguousarray(tifffile.imread(source)).tobytes()).hexdigest()
    assert entry["declared_spacing"] == {"xy": D.VOXEL_XY, "z": D.VOXEL_Z} and entry["renormalised"] is False
    # A file not of sum 1 is renormalised in memory, and the record says so.
    tifffile.imwrite(source, D.psf_small() * 2)
    manifest, path, copy, statuses = stage(work, "ps2", block(work, rounds=["round1"]), dw, kinds=("psf",))
    entry = json.loads((copy / "psf" / "psf.json").read_text())["labels"]["ch00"]
    assert entry["renormalised"] is True and entry["sum"] == pytest.approx(2.0, abs=1e-6)
    assert float(tifffile.imread(copy / "psf" / "ch00.tif").astype(np.float64).sum()) == pytest.approx(1.0, abs=1e-6)
    # A negative value, or a file that is not ZYX, fails the psf task.
    for bad in (D.psf_small() - 0.01, D.psf_small()[3]):
        tifffile.imwrite(source, bad)
        manifest, path, copy, statuses = stage(work, "ps2", block(work, rounds=["round1"]), dw, kinds=("psf",))
        assert statuses["psf"] == 1
        assert "PSF file" in failed_record(path, "psf")["error"]["message"]


# --- PS3, PS4: generation (psfmodels) -------------------------------------------------------------------------------

W855 = {"na": 1.30, "immersion_ri": 1.518, "sample_ri": 1.338, "pinhole_radius_nm": 307, "depth_um": 0.0,
        "shape_zyx": [21, 33, 33], "channels": {"ch00": {"excitation_nm": 499, "emission_nm": 529}}}


def generated(dxy, dz=0.346255):
    pytest.importorskip("psfmodels", reason="PS3, PS4 and RU11 need psfmodels: the deconvolution extra "
                                             "(uv sync --extra deconvolution), which this environment lacks")
    from starfinder.workflow._deconvolve import generate_psf, psf_parameters
    parameters = psf_parameters(W855, "ch00", dxy, dz)
    array, rec, version = generate_psf(parameters)
    return array, rec, parameters


def test_ps3_generation():
    array, rec, parameters = generated(0.094635)
    assert array.shape == (21, 33, 33) and rec["peak_after"] == [10, 16, 16] and rec["shift_zyx"] == [0, 0, 0]
    assert float(array.astype(np.float64).sum()) == pytest.approx(1.0, abs=1e-6)
    assert fwhm(array[:, 16, 16], 0.346255) == pytest.approx(0.4933, abs=1e-4)
    assert fwhm(array[10, 16, :], 0.094635) == pytest.approx(0.2362, abs=1e-4)
    assert parameters["dxy"] == 0.094635 and parameters["dz"] == 0.346255


def test_ps4_the_spacing_reaches_the_psf():
    first, _, _ = generated(0.094635)
    second, _, parameters = generated(0.0473175)
    assert parameters["dxy"] == 0.0473175
    assert hashlib.sha256(first.tobytes()).hexdigest() != hashlib.sha256(second.tobytes()).hexdigest()
    voxels = [fwhm(a[10, 16, :], 1.0) for a in (first, second)]
    assert abs(voxels[1] - 2 * voxels[0]) <= 1.0


# --- WI1 to WI7: FFTW wisdom ----------------------------------------------------------------------------------------

IDENTITY = {"version": "0.4.5", "sha256": "a" * 64, "fft_backend": "fftw-3.3.8-sse2-avx"}


@pytest.mark.contract
def test_wi1_the_key():
    base = dict(image_shape=(8, 32, 32), psf_shape=(7, 9, 9), border_quality=2, tile_size=None, threads=2,
                cpu="c" * 64, identity=IDENTITY)
    fields = wisdom_fields(**base)
    assert fields["job_shape_zyx"] == [14, 40, 40]
    key = wisdom_key(fields)
    assert key == hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for name, value in [("image_shape", (8, 32, 33)), ("psf_shape", (7, 9, 11)), ("border_quality", 1),
                        ("tile_size", 64), ("threads", 4), ("cpu", "d" * 64),
                        ("identity", dict(IDENTITY, fft_backend="fftw-3.3.10")),
                        ("identity", dict(IDENTITY, version="0.4.6")), ("identity", dict(IDENTITY, sha256="b" * 64))]:
        assert wisdom_key(wisdom_fields(**dict(base, **{name: value}))) != key, name
    # Two PSFs of one shape and different values give one key: the key reads the PSF's shape only.
    small, core = D.psf_small(), D.psf_core()
    assert not np.array_equal(small, core)
    assert wisdom_key(wisdom_fields(**dict(base, psf_shape=small.shape))) == \
        wisdom_key(wisdom_fields(**dict(base, psf_shape=core.shape))) == key


def store_for(work, dw, threads, *, matching=True, damaged=False):
    """``wisdom_store``: a manifest with the key the task computes (or another) and one other key."""
    store = work / "store"
    identity = deconwolf_identity(str(dw))
    fields = wisdom_fields(D.SHAPE, (7, 9, 9), 2, None, threads, cpu_identity()["sha256"], identity)
    keys = {wisdom_key(fields) if matching else "f" * 64: fields, "e" * 64: dict(fields, threads=99)}
    manifest = {"format": "starfinder.deconwolf-wisdom/1", "keys": {}}
    for key, value in keys.items():
        file = store / key / wisdom_name(value["threads"])
        file.parent.mkdir(parents=True)
        file.write_text(f"wisdom of {key}\n")
        manifest["keys"][key] = {"fields": value, "file": f"{key}/{file.name}", "bytes": file.stat().st_size,
                                 "sha256": D.sha256(file)}
        if damaged:
            file.write_text("damaged\n")
    (store / "manifest.json").write_text(json.dumps(manifest))
    return store, wisdom_key(fields)


def run_with_store(work, store, name, threads=2):
    dw, calls = D.stand_in(work / "bin")
    execution = {"tasks": {"fov": {"threads": 1}, "deconvolve": {"threads": threads}}}
    if store is not None:
        execution["deconwolf_wisdom"] = str(store)
    manifest, path, copy, statuses = stage(work, name, block(work, rounds=["round1"]), dw, execution=execution)
    return manifest, path, copy, statuses, D.calls(calls)


def test_wi2_a_matching_key(work, tmp_path):
    dw, _ = D.stand_in(work / "bin")
    store, key = store_for(work, dw, 2)
    before = D.tree(store)
    manifest, path, copy, statuses, calls = run_with_store(work, store, "wi2")
    assert statuses[deconvolve_id("round1")] == 0 and len(calls) == 2
    import tempfile
    for call in calls:
        home = Path(call["HOME"])
        assert str(home).startswith(tempfile.gettempdir()) and not home.exists()
        assert call["cwd"] == str(home) and "--noplan" not in call["argv"]
        assert call["wisdom_found"] == [str(home / ".config" / "deconwolf" / wisdom_name(2))]
    # The first call finds the store's file; a later one finds it as deconwolf wrote it back in the private folder.
    assert set(calls[0]["wisdom_found_sha256"].values()) == {D.sha256(store / key / wisdom_name(2))}
    planning = record(copy, "round1")["planning"]
    assert planning["mode"] == "measure" and planning["wisdom_key"] == key
    assert planning["wisdom_sha256"] == D.sha256(store / key / wisdom_name(2))
    assert D.tree(store) == before


@pytest.mark.parametrize("case", ["no matching key", "no store"])
def test_wi3_no_match_or_no_store(work, case):
    dw, _ = D.stand_in(work / "bin")
    store = store_for(work, dw, 2, matching=False)[0] if case == "no matching key" else None
    manifest, path, copy, statuses, calls = run_with_store(work, store, "wi3")
    assert all("--noplan" in call["argv"] and call["wisdom_found"] == [] for call in calls)
    planning = record(copy, "round1")["planning"]
    assert planning["mode"] == "estimate" and planning["reason"] == case


def test_wi4_a_damaged_store(work):
    dw, _ = D.stand_in(work / "bin")
    store, key = store_for(work, dw, 2, damaged=True)
    manifest, path, copy, statuses, calls = run_with_store(work, store, "wi4")
    assert statuses[deconvolve_id("round1")] == 1
    assert str(store / key / wisdom_name(2)) in failed_record(path, deconvolve_id("round1"))["error"]["message"]


def test_wi5_nothing_written_outside_the_task(work, tmp_path, monkeypatch):
    dw, _ = D.stand_in(work / "bin")
    store, key = store_for(work, dw, 2)
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    (home / ".config").mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    task_tmp = tmp_path / "task-tmp"
    task_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(task_tmp))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    config = D.configuration(work, "wi5", deconvolution=block(work, rounds=["round1"]), dw=dw, selected_fovs=[FOV],
                             execution={"tasks": {"deconvolve": {"threads": 2}}, "deconwolf_wisdom": str(store)})
    manifest, path = D.plan(work, "wi5", config)
    assert D.run(path, "psf") == 0
    before = {"store": D.tree(store), "home": D.tree(home), "cwd": D.tree(cwd), "input": D.tree(work / "input")}
    copy = D.copy_folder(manifest)
    copy_before = D.tree(copy)
    assert D.run(path, deconvolve_id("round1")) == 0
    after = {"store": D.tree(store), "home": D.tree(home), "cwd": D.tree(cwd), "input": D.tree(work / "input")}
    assert after == before and list(task_tmp.iterdir()) == []
    declared = {o for t in manifest["tasks"] if t["id"] == deconvolve_id("round1") for o in t["outputs"]}
    new = {f"deconvolved/{SAMPLE}/{copy.name}/{name}" for name, value in D.tree(copy).items()
           if copy_before.get(name) != value}
    assert new == declared


@pytest.mark.slow
def test_wi6_one_generating_job_per_key(work, tmp_path):
    dw, calls = D.stand_in(work / "bin", delay=1.0)
    store = tmp_path / "store"
    command = [sys.executable, "-m", "starfinder", "deconvolve", "wisdom", "--store", str(store), "--shape", "8",
               "32", "32", "--psf-shape", "7", "9", "9", "--threads", "2", "--deconwolf", str(dw)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    key = json.loads(first.stdout)["key"]
    manifest = json.loads((store / "manifest.json").read_text())
    assert list(manifest["keys"]) == [key]
    assert D.sha256(store / manifest["keys"][key]["file"]) == manifest["keys"][key]["sha256"]
    before = D.tree(store)
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 1 and "already holds the key" in second.stderr and D.tree(store) == before
    # Two concurrent calls for a new key: one entry, one file, and the manifest parses.
    other = tmp_path / "other"
    started = [subprocess.Popen([*command[:6], str(other), *command[7:]], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for _ in range(2)]
    statuses = sorted(p.communicate() and p.returncode for p in started)
    assert statuses == [0, 1]
    manifest = json.loads((other / "manifest.json").read_text())
    assert len(manifest["keys"]) == 1
    assert sorted(p.relative_to(other).as_posix() for p in other.rglob("*") if p.is_file()) == sorted(
        ["manifest.json", manifest["keys"][next(iter(manifest["keys"]))]["file"]])


def test_wi7_gpu_tasks(work):
    dw, calls = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "wi7", block(work, rounds=["round1"]), dw,
                                           execution={"tasks": {"deconvolve": {"gpu": True}}})
    for call in D.calls(calls):
        assert "--gpu" in call["argv"] and call["argv"][call["argv"].index("--threads") + 1] == "1"
        assert call["wisdom_found"] == []
    rec = record(copy, "round1")
    assert rec["planning"] == {"mode": "gpu"} and rec["device"]["kind"] == "gpu"


# --- RC1 to RC5: records ------------------------------------------------------------------------------------------

def test_rc1_rc2_the_round_record_and_copy_json(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "rc1", block(work, rounds=["round1"]), dw)
    rec = record(copy, "round1")
    assert set(rec) >= {"format", "copy_id", "fov", "round", "labels", "image_dir", "deconwolf", "device", "options",
                        "storage", "planning", "channels", "code", "started_at", "ended_at", "host", "job_id"}
    assert rec["format"] == "starfinder.deconvolution/1" and rec["round"] == "round1" and rec["fov"] == FOV
    assert rec["deconwolf"]["sha256"] == D.sha256(dw) and rec["deconwolf"]["version"] == "0.4.5"
    psf_record = json.loads((copy / "psf" / "psf.json").read_text())
    for channel in rec["channels"]:
        label = channel["label"]
        source = raw(work, "round1", label)
        x = tifffile.imread(source)
        assert channel["input"]["file_sha256"] == D.sha256(source)
        assert channel["input"]["sha256"] == hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        out = copy / channel["output"]["path"]
        assert channel["output"]["file_sha256"] == D.sha256(out)
        assert channel["output"]["sha256"] == hashlib.sha256(tifffile.imread(out).tobytes()).hexdigest()
        assert channel["psf"]["file_sha256"] == psf_record["labels"][label]["copy"]["file_sha256"]
        assert channel["iterations_done"] == 3
        assert channel["log"]["sha256"] == D.sha256(copy / "records" / "round1" / FOV / channel["log"]["path"])
        for key in ("float_min", "float_max", "kept_offset", "clipped_low", "clipped_high", "wall_s", "cpu_s"):
            assert key in channel
    for key in ("iterations", "method", "border_quality", "tile_size", "tile_padding", "offset", "background"):
        assert key in rec["options"]
    # RC2: copy.json holds the id and its seven fields, and psf.json's SHA-256; the id recomputes.
    copied = json.loads((copy / "copy.json").read_text())
    assert set(copied["fields"]) == {"dataset_id", "sample_id", "image_dir_template", "voxel_size_xy",
                                     "voxel_size_z", "deconvolution", "psf_files"}
    assert "copy_root" not in copied["fields"]["deconvolution"]
    assert copied["psf"]["sha256"] == D.sha256(copy / "psf" / "psf.json")
    text = json.dumps(copied["fields"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(text.encode()).hexdigest()[:12] == copied["copy_id"] == rec["copy_id"]
    assert copied["fields"]["psf_files"]["ch00"] == D.sha256(psf_files(work)["ch00"])


def test_rc3_run_json_names_the_copy(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "rc3", block(work, rounds=[*D.ROUNDS, "reference_stain"]), dw,
                                           kinds=("psf", "deconvolve", "fov", "morphology"), morphology=True)
    assert set(statuses.values()) == {0}, statuses
    out = path.parents[1]
    run_json = json.loads((out / "fovs" / FOV / "run.json").read_text())
    entry = run_json["deconvolution"]
    assert entry["copy"] == str(copy.absolute()) and entry["copy_id"] == manifest["deconvolution"]["copy_id"]
    assert sorted(entry["rounds"]) == list(D.ROUNDS)
    path1 = copy / "records" / "round1" / FOV / "deconvolution.json"
    assert entry["rounds"]["round1"] == {"record": str(path1), "record_sha256": D.sha256(path1)}
    outputs = {str(copy / c["output"]["path"]): c["output"]["file_sha256"]
               for name in D.ROUNDS for c in record(copy, name)["channels"]}
    images = [i for i in run_json["inputs"] if i["path"].endswith(".tif")]
    assert {i["path"]: i["sha256"] for i in images} == outputs
    morph = json.loads((out / "fovs" / FOV / "nuclei_registration.json").read_text())["deconvolution"]
    stain = copy / "records" / "reference_stain" / FOV / "deconvolution.json"
    assert morph["rounds"] == {"reference_stain": {"record": str(stain), "record_sha256": D.sha256(stain)}}


def test_rc4_completion_and_failure(work):
    dw, _ = D.stand_in(work / "bin", fail_on="ch01")
    manifest, path, copy, statuses = stage(work, "rc4", block(work, rounds=["round1"]), dw)
    task = deconvolve_id("round1")
    assert statuses[task] == 1
    failed = failed_record(path, task)
    assert failed["status"] == "failed" and (path.parents[1] / failed["log"]["path"]).is_file()
    assert not list(copy.glob("round1/**/*.tif")) and not (copy / "records" / "round1" / FOV / "deconvolution.json").exists()
    assert not (path.parents[1] / "workflow" / "tasks" / f"{task.replace('/', '.')}.json").exists()
    good, _ = D.stand_in(work / "good")
    manifest, path, copy, statuses = stage(work, "rc4", block(work, rounds=["round1"]), good)
    listed = {o["path"]: o["sha256"] for o in task_record(path, task)["outputs"]}
    declared = next(t for t in manifest["tasks"] if t["id"] == task)["outputs"]
    for output in declared:
        assert listed[output] == "sha256:" + D.sha256(Path(output) if Path(output).is_absolute()
                                                      else path.parents[1] / output)


def run_together(path, tasks):
    """Start the commands of ``tasks`` at once, each in its own process; their exit statuses."""
    command = [sys.executable, "-m", "starfinder", "workflow", "deconvolve", "--manifest", str(path), "--task"]
    started = [subprocess.Popen([*command, task], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for task in tasks]
    return [p.communicate() and p.returncode for p in started]


@pytest.mark.parametrize("order", ["reference first", "stain first", "together"])
def test_rc5_a_shared_image_folder_keeps_two_records(work, order):
    dw, calls = D.stand_in(work / "bin")
    config = D.configuration(work, "rc5", deconvolution=block(work, rounds=["round1", "reference_stain"]), dw=dw,
                             selected_fovs=[FOV], morphology=True)
    manifest, path = D.plan(work, "rc5", config)
    assert D.run(path, "psf") == 0
    pair = [deconvolve_id("round1"), deconvolve_id("reference_stain")]
    if order == "together":
        assert run_together(path, pair) == [0, 0]
    else:
        for task in (pair if order == "reference first" else pair[::-1]):
            assert D.run(path, task) == 0
    copy = D.copy_folder(manifest)
    for name, labels in (("round1", ["ch00", "ch01"]), ("reference_stain", ["dapi"])):
        rec = record(copy, name)
        assert rec["labels"] == labels
        for channel in rec["channels"]:
            assert channel["log"]["sha256"] == D.sha256(copy / "records" / name / FOV / channel["log"]["path"])


def test_rc5_a_rewritten_raw_file_redoes_its_round_only(work):
    dw, calls = D.stand_in(work / "bin")
    config = D.configuration(work, "rc5b", deconvolution=block(work, rounds=["round1", "reference_stain"]), dw=dw,
                             selected_fovs=[FOV], morphology=True)
    manifest, path = D.plan(work, "rc5b", config)
    D.run_kinds(manifest, path, "psf", "deconvolve")
    copy = D.copy_folder(manifest)
    for rewritten, kept in (("reference_stain", "round1"), ("round1", "reference_stain")):
        label = "dapi" if rewritten == "reference_stain" else "ch00"
        tifffile.imwrite(raw(work, rewritten, label), D.spots(99 + len(rewritten)))
        images = (f"round1/{FOV}/ch00.tif", f"round1/{FOV}/ch01.tif") if kept == "round1" else (
            f"round1/{FOV}/dapi.tif",)
        kept_files = {k: v for k, v in D.tree(copy).items() if k in images or k.startswith(f"records/{kept}/")}
        before = len(D.calls(calls))
        manifest, path = D.plan(work, "rc5b", config)
        assert D.run_kinds(manifest, path, "deconvolve") == {
            deconvolve_id("round1"): 0, deconvolve_id("reference_stain"): 0}
        assert task_record(path, deconvolve_id(rewritten))["reused"] is False
        assert task_record(path, deconvolve_id(kept))["reused"] is True
        assert len(D.calls(calls)) - before == (1 if rewritten == "reference_stain" else 2)
        after = D.tree(copy)
        assert {k: after[k] for k in kept_files} == kept_files
        verify = subprocess.run([sys.executable, "-m", "starfinder", "workflow", "verify", "--manifest", str(path),
                                 "--task", deconvolve_id(kept)], capture_output=True, text=True)
        assert verify.returncode == 0 and "\tcomplete" in verify.stdout
    # A failure of the stain task removes only dapi.tif and the stain's record folder.
    failing, _ = D.stand_in(work / "failing", fail_on="dapi")
    config["execution"]["deconwolf"] = str(failing)
    tifffile.imwrite(raw(work, "reference_stain", "dapi"), D.spots(7))
    manifest, path = D.plan(work, "rc5b", config)
    before = D.tree(copy)
    assert D.run(path, deconvolve_id("reference_stain")) == 1
    after = D.tree(copy)
    gone = set(before) - set(after)
    assert gone == {f"round1/{FOV}/dapi.tif", f"records/reference_stain/{FOV}/deconvolution.json",
                    f"records/reference_stain/{FOV}/dapi.dw.log"}
    assert {k: after[k] for k in after} == {k: before[k] for k in after}


# --- RU1 to RU11: reuse and staleness -------------------------------------------------------------------------------

def ru_config(work, output_id, dw, **keys):
    return D.configuration(work, output_id, deconvolution=block(work, copy_root=str(work / "copies"),
                                                                rounds=list(D.ROUNDS), **keys),
                           dw=dw, selected_fovs=[FOV])


def run_stage(work, name, config):
    manifest, path = D.plan(work, name, config)
    statuses = D.run_kinds(manifest, path, "psf", "deconvolve")
    assert set(statuses.values()) == {0}, statuses
    return manifest, path, D.copy_folder(manifest)


def test_ru1_ru2_ru3_verified_reuse(work):
    dw, calls = D.stand_in(work / "bin")
    first, path, copy = run_stage(work, "ru1-a", ru_config(work, "ru1-a", dw))
    assert len(D.calls(calls)) == 4
    before = D.tree(copy)
    second, path, copy2 = run_stage(work, "ru1-b", ru_config(work, "ru1-b", dw))
    assert copy2 == copy and len(D.calls(calls)) == 4 and D.tree(copy) == before
    assert task_record(path, "psf")["reused"] is True
    assert all(task_record(path, deconvolve_id(name))["reused"] is True for name in D.ROUNDS)
    # RU2: one raw channel rewritten: that round is redone, the other reused.
    tifffile.imwrite(raw(work, "round2", "ch01"), D.spots(123))
    third, path, _ = run_stage(work, "ru1-c", ru_config(work, "ru1-c", dw))
    assert len(D.calls(calls)) == 6
    assert task_record(path, deconvolve_id("round2"))["reused"] is False
    assert task_record(path, deconvolve_id("round1"))["reused"] is True
    after = D.tree(copy)
    assert after[f"records/round2/{FOV}/deconvolution.json"] != before[f"records/round2/{FOV}/deconvolution.json"]
    assert {k: v for k, v in after.items() if "round1" in k} == {k: v for k, v in before.items() if "round1" in k}
    # RU3: another iterations count: a new copy id and folder; the first copy is unchanged.
    kept = D.tree(copy)
    fourth, path, copy4 = run_stage(work, "ru1-d", ru_config(work, "ru1-d", dw, iterations=4))
    assert copy4 != copy and D.tree(copy) == kept and len(D.calls(calls)) == 10


def test_ru4_a_consumer_checks(work):
    dw, _ = D.stand_in(work / "bin")
    manifest, path, copy, statuses = stage(work, "ru4", block(work), dw, kinds=("psf", "deconvolve"))
    target = copy / "round1" / FOV / "ch00.tif"
    recorded = D.sha256(target)
    tifffile.imwrite(target, np.zeros(D.SHAPE, dtype=np.uint8))
    assert D.run(path, f"fov/{SAMPLE}/{FOV}") == 1
    message = failed_record(path, f"fov/{SAMPLE}/{FOV}")["error"]["message"]
    assert str(target) in message and recorded in message and D.sha256(target) in message


def test_ru5_resume_and_saved_sets(work):
    """A verified resume after the copy was redone with other values is refused on the input hashes; a saved set made
    without the block is refused by a run with it, naming both copy ids; a set and a run of one copy id are accepted."""
    dw, _ = D.stand_in(work / "bin")
    config = D.configuration(work, "ru5", deconvolution=block(work), dw=dw, selected_fovs=[FOV])
    config["processing"]["checkpoints"] = ["registered", "candidates"]
    config["execution"]["resume_from_checkpoint"] = True
    manifest, path = D.plan(work, "ru5", config)
    assert set(D.run_kinds(manifest, path).values()) == {0}
    redone, _ = D.stand_in(work / "redone", shift=1)
    config["execution"]["deconwolf"] = str(redone)
    copy = D.copy_folder(manifest)
    for target in (copy / "records").rglob("deconvolution.json"):
        target.unlink()
    manifest, path = D.plan(work, "ru5", config)
    assert set(D.run_kinds(manifest, path, "deconvolve").values()) == {0}
    assert D.run(path, f"fov/{SAMPLE}/{FOV}") == 3
    # The saved sets: one made without the block, one with it.
    plain = D.configuration(work, "set-plain", selected_fovs=[FOV])
    plain["processing"]["checkpoints"] = ["registered"]
    manifest, path = D.plan(work, "set-plain", plain)
    assert set(D.run_kinds(manifest, path).values()) == {0}
    with_block = D.configuration(work, "set-dw", deconvolution=block(work), dw=dw, selected_fovs=[FOV])
    with_block["processing"]["checkpoints"] = ["registered"]
    manifest, path = D.plan(work, "set-dw", with_block)
    assert set(D.run_kinds(manifest, path).values()) == {0}
    copy_id = manifest["deconvolution"]["copy_id"]
    for source, expected in (("set-plain", 2), ("set-dw", 0)):
        reader = D.configuration(work, f"read-{source}", deconvolution=block(work), dw=dw, selected_fovs=[FOV])
        reader["processing"]["spot_finding_input"] = str(work / "output" / D.DATASET_ID / source)
        manifest, path = D.plan(work, f"read-{source}", reader)
        assert manifest["deconvolution"]["copy_id"] == copy_id
        statuses = D.run_kinds(manifest, path, "fov")
        assert set(statuses.values()) == {expected}, statuses


def test_ru6_one_writer_per_round(work):
    dw, calls = D.stand_in(work / "bin", delay=0.5)
    paths = []
    for name in ("ru6-a", "ru6-b"):
        config = ru_config(work, name, dw)
        config["deconvolution"]["rounds"] = ["round1"]
        manifest, path = D.plan(work, name, config)
        assert D.run(path, "psf") == 0
        paths.append(path)
    command = [sys.executable, "-m", "starfinder", "workflow", "deconvolve", "--task", deconvolve_id("round1"),
               "--manifest"]
    started = [subprocess.Popen([*command, str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for p in paths]
    assert [p.communicate() and p.returncode for p in started] == [0, 0]
    assert len(D.calls(calls)) == len(D.LABELS)
    reused = sorted(task_record(p, deconvolve_id("round1"))["reused"] for p in paths)
    assert reused == [False, True]


def test_ru7_code_version_rerun(work):
    dw, calls = D.stand_in(work / "bin")
    run_stage(work, "ru7-a", ru_config(work, "ru7-a", dw))
    other, other_calls = D.stand_in(work / "other", name="dw")
    with open(other, "a") as handle:
        handle.write("# another build\n")
    config = ru_config(work, "ru7-b", other)
    manifest, path, copy = run_stage(work, "ru7-b", config)
    assert D.calls(other_calls) == []
    config["execution"]["code_version"] = "rerun"
    manifest, path, copy = run_stage(work, "ru7-c", dict(config, output_id="ru7-c"))
    assert len(D.calls(other_calls)) == 4
    assert all(task_record(path, deconvolve_id(name))["reused"] is False for name in D.ROUNDS)


@pytest.mark.parametrize("key", ["voxel_size_xy", "voxel_size_z"])
def test_ru8_a_change_of_spacing_alone(work, key):
    dw, calls = D.stand_in(work / "bin")
    base = ru_config(work, "ru8-a", dw)
    first, path, copy = run_stage(work, "ru8-a", base)
    kept = D.tree(copy)
    changed = json.loads(json.dumps(base))
    changed["output_id"] = "ru8-b"
    changed[key] = base[key] * 1.1
    changed["deconvolution"]["psf"]["voxel_size_um"]["xy" if key == "voxel_size_xy" else "z"] = changed[key]
    second, path, copy2 = run_stage(work, "ru8-b", changed)
    assert copy2 != copy and len(D.calls(calls)) == 8 and D.tree(copy) == kept
    spacing = json.loads((copy2 / "psf" / "psf.json").read_text())["labels"]["ch00"]["declared_spacing"]
    assert spacing["xy" if key == "voxel_size_xy" else "z"] == changed[key]
    fourth, path, copy4 = run_stage(work, "ru8-c", dict(base, output_id="ru8-c"))
    assert copy4 == copy and len(D.calls(calls)) == 8 and D.tree(copy) == kept


def test_ru9_a_changed_supplied_psf(work):
    dw, calls = D.stand_in(work / "bin")
    first, path, copy = run_stage(work, "ru9-a", ru_config(work, "ru9-a", dw))
    kept = D.tree(copy)
    D.write_psf_files(work / "psf", array=D.psf_core())
    second, path, copy2 = run_stage(work, "ru9-b", ru_config(work, "ru9-b", dw))
    assert copy2 != copy and len(D.calls(calls)) == 8 and D.tree(copy) == kept
    assert D.sha256(copy2 / "psf" / "ch00.tif") != D.sha256(copy / "psf" / "ch00.tif")


@pytest.mark.parametrize("damage", ["file", "field"])
def test_ru10_a_damaged_copys_psf(work, capsys, damage):
    dw, _ = D.stand_in(work / "bin")
    first, path, copy = run_stage(work, "ru10-a", ru_config(work, "ru10-a", dw))
    if damage == "file":
        tifffile.imwrite(copy / "psf" / "ch00.tif", D.psf_core())
    else:
        record_path = copy / "psf" / "psf.json"
        data = json.loads(record_path.read_text())
        data["labels"]["ch00"]["declared_spacing"]["xy"] = 0.2
        record_path.write_text(json.dumps(data, indent=1))
    before = D.tree(copy)
    manifest, path = D.plan(work, "ru10-b", ru_config(work, "ru10-b", dw))
    capsys.readouterr()
    assert D.run(path, "psf") == 2
    message = capsys.readouterr().err
    assert ("ch00.tif" in message) if damage == "file" else ("labels.ch00.declared_spacing" in message)
    assert D.tree(copy) == before


def test_ru11_the_spacing_reaches_a_generated_psf(work):
    pytest.importorskip("psfmodels", reason="PS3, PS4 and RU11 need psfmodels: the deconvolution extra "
                                             "(uv sync --extra deconvolution), which this environment lacks")
    dw, calls = D.stand_in(work / "bin")
    base = D.configuration(work, "ru11-a", deconvolution=D.block(copy_root=str(work / "copies"),
                                                                 rounds=list(D.ROUNDS)), dw=dw, selected_fovs=[FOV])
    first, path, copy = run_stage(work, "ru11-a", base)
    changed = dict(base, output_id="ru11-b", voxel_size_xy=base["voxel_size_xy"] * 1.1)
    second, path, copy2 = run_stage(work, "ru11-b", changed)
    assert copy2 != copy and D.sha256(copy2 / "psf" / "ch00.tif") != D.sha256(copy / "psf" / "ch00.tif")
    assert json.loads((copy2 / "psf" / "psf.json").read_text())["labels"]["ch00"]["parameters"]["dxy"] == \
        changed["voxel_size_xy"]


# --- DW1 to DW3: deconwolf itself ---------------------------------------------------------------------------------

@pytest.fixture
def real_dw():
    path = os.environ.get("STARFINDER_TEST_DECONWOLF") or shutil.which("dw")
    if not path:
        pytest.skip("DW1 to DW3 need the deconwolf binary: dw on PATH or STARFINDER_TEST_DECONWOLF, which this host "
                    "lacks (the image has it)")
    return path


def test_dw1_the_options_exist(real_dw):
    identity = deconwolf_identity(real_dw)
    assert identity["version"] == "0.4.5"
    text = subprocess.run([real_dw, "--help"], capture_output=True, text=True)
    for option in ("--iter", "--threads", "--gpu", "--tilesize", "--noplan", "--offset", "--out", "--float"):
        assert option in text.stdout + text.stderr, option


@pytest.mark.slow
def test_dw2_the_offset_rule_and_determinism(real_dw, tmp_path):
    from scipy.ndimage import convolve
    image = np.zeros((16, 64, 64), dtype=np.float64)
    for z, y, x, value in ((8, 20, 20, 1000.0), (6, 40, 44, 600.0), (10, 30, 12, 400.0)):
        image[z, y, x] = value
    blurred = convolve(image, D.psf_small().astype(np.float64), mode="constant").astype(np.float32)
    tifffile.imwrite(tmp_path / "image.tif", blurred)
    tifffile.imwrite(tmp_path / "psf.tif", D.psf_small())
    home = tmp_path / "home"
    (home / ".config" / "deconwolf").mkdir(parents=True)

    def run(name, *extra, threads=1):
        out = tmp_path / f"{name}.tif"
        argv = [real_dw, "--iter", "3", "--threads", str(threads), "--noplan", "--float", *extra, "--out", str(out),
                str(tmp_path / "image.tif"), str(tmp_path / "psf.tif")]
        done = subprocess.run(argv, cwd=home, env=dict(os.environ, HOME=str(home), OMP_NUM_THREADS=str(threads)),
                              capture_output=True)
        assert done.returncode == 0, done.stderr
        return out
    one, two = run("one"), run("two", threads=2)
    assert abs(float(tifffile.imread(one).min()) - 5) <= 0.5
    assert D.sha256(one) == D.sha256(two)
    assert float(tifffile.imread(run("tiled", "--tilesize", "32")).min()) < 0.5
    assert float(tifffile.imread(one)[8, 20, 20]) - 5 > float(blurred[8, 20, 20])


@pytest.mark.slow
def test_dw3_the_stage_on_the_real_binary(real_dw, work):
    manifest, path, copy, statuses = stage(work, "dw3", block(work, rounds=["round1"]), real_dw)
    assert set(statuses.values()) == {0}
    rec = record(copy, "round1")
    assert rec["planning"]["mode"] == "estimate" and rec["deconwolf"]["sha256"] == D.sha256(real_dw)
    for channel in rec["channels"]:
        assert channel["iterations_done"] == 3
        assert tags(copy / channel["output"]["path"])[:3] == ({8}, False, "ZYX")
