"""Generator checks of the appearance bank (docs/synthetic-real-appearance-validation.md, "Generator checks").

GA3 (the bank's refusals), GA4 (the build is deterministic and recorded), GA5 (the selection rules), GA6 (what an
entry carries), GA14 (brightness: the bank's part) and GA22 (the closed-form bank reads no real data: the bank's
part), on the fixtures ``bank_cases``, ``fall_off_cases``, ``cf_banks`` and ``cf_bank_files``
(``cellular_appearance_scenes.py``), recomputed by the test oracle (``cellular_appearance_oracle.py``). Every bank is
built in the session from hand-made arrays or from scenes of the accepted closed-form generator; files are written
under the test's temporary directories only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi
import tifffile

from starfinder.synthetic import BankReference, build_appearance_bank, context_scene_preset, load_appearance_bank

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import (HELD_OUT, ROUTINE, SEEDS, bank_cases, cf_bank, cf_bank_files, cf_sources,
                                         fall_off_cases, write_case, write_config)

pytestmark = [pytest.mark.synthetic]

# The file banks of the dense 3D and culture presets take 5 s or more to set up on one CPU (four scenes generated and
# written by save_context_scene); the 2D tissue one stays below.
FILE_PRESETS = ["context-tissue2d-v1", pytest.param("context-dense3d-v1", marks=pytest.mark.slow),
                pytest.param("context-culture-v1", marks=pytest.mark.slow)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key != "record"}


def built(config, directory, given=None):
    """Build ``config`` (or ``given``, its JSON file) into ``directory/<bank>.npz``; return (record, path)."""
    path = Path(directory) / f"{config['bank']}.npz"
    return build_appearance_bank(config if given is None else given, path), path


@pytest.fixture(scope="session")
def file_banks(tmp_path_factory):
    """``cf_bank_files`` per preset, built twice into two files (GA4); cached for the session."""
    cache = {}

    def get(preset):
        if preset not in cache:
            root = tmp_path_factory.mktemp(preset)
            config = cf_bank_files(preset, root / "scenes")
            first = built(config, root / "one")
            second = built(config, root / "two", write_config(config, root / "config.json"))
            cache[preset] = (root, config, first, second)
        return cache[preset]
    return get


def configured_files(config):
    """Every file a build configuration names: the sources' (stains, labels, nucleus labels) and the held-out ones."""
    sources = set()
    for source in config["sources"]:
        for spec in (*source["stains"].values(), source["labels"], source.get("nucleus_labels")):
            if spec is not None:
                sources.add(str(Path(spec["path"] if isinstance(spec, dict) else spec).resolve()))
    held = {str(Path(f).resolve()) for h in config["held_out"] for f in h["files"]}
    return sources, held


# ---------------------------------------------------------------------------------------------------- GA4

@pytest.mark.validation
@pytest.mark.parametrize("preset", ROUTINE)
def test_closed_form_bank_build_is_deterministic_and_recorded(preset):
    from starfinder.synthetic._appearance_bank import _closed_form_bank, _digest
    bank = cf_bank(preset)
    again = _closed_form_bank.__wrapped__(preset, 4)
    assert sorted(bank.arrays) == sorted(again.arrays)
    for key in bank.arrays:
        assert bank.arrays[key].dtype == again.arrays[key].dtype, key
        assert np.array_equal(bank.arrays[key], again.arrays[key]), key
    assert bank.digest == again.digest == _digest(bank.arrays) == bank.record["digest"]
    record = bank.record
    sources = [f"{preset}/seed-{seed}" for seed in SEEDS if seed != HELD_OUT]
    assert [(i["image"], i["role"], i["preset"], i["seed"]) for i in record["inputs"]] == [
        (f"{preset}/seed-{seed}", "source", preset, seed) for seed in SEEDS if seed != HELD_OUT]
    assert not any("sha256" in i or "path" in i for i in record["inputs"])
    assert record["held_out"] == [dict(image=f"{preset}/seed-{HELD_OUT}",
                                       inputs=[dict(role="held_out", preset=preset, seed=HELD_OUT)])]
    assert (record["preset"], record["n"], record["seeds"]) == (preset, 4, list(SEEDS))
    assert [r["image"] for r in record["images"]] == sources
    for row in record["images"]:
        for key in ("labels", "left_out_border", "left_out_border_z_alone", "left_out_size_after_border",
                    "left_out_nucleus_after_size", "left_out_squashed_after_rest", "kept", "plateau",
                    "visible_median", "patches"):
            assert key in row, key
        assert row["labels"] == sum(row[k] for k in ("left_out_border", "left_out_size_after_border",
                                                       "left_out_nucleus_after_size", "left_out_squashed_after_rest",
                                                       "kept"))
    assert record["counts"]["entries"] == len(bank) == sum(r["kept"] for r in record["images"])
    assert record["configuration"]["selection"]["rim_share_max"] == (0.35 if "culture" in preset else 0.25)


@pytest.mark.validation
@pytest.mark.parametrize("preset", FILE_PRESETS)
def test_file_bank_build_is_deterministic_and_recorded(file_banks, preset):
    root, config, (record, path), (record2, path2) = file_banks(preset)
    first, second = npz_arrays(path), npz_arrays(path2)
    assert sorted(first) == sorted(second)
    for key in first:
        assert first[key].dtype == second[key].dtype and np.array_equal(first[key], second[key]), key
    assert record["digest"] == record2["digest"]
    # The same sources give the same bank as the closed-form build in memory (CF1 and the file route agree).
    assert record["digest"] == cf_bank(preset).digest
    for rec, file in ((record, path), (record2, path2)):
        assert rec["file"]["sha256"] == sha256(file) and rec["file"]["size"] == file.stat().st_size
        assert json.loads(file.with_name(f"{config['bank']}.record.json").read_text()) == rec
    sources, held = configured_files(config)
    listed = {(i["path"], i["sha256"]) for i in record["inputs"]}
    assert listed == {(p, sha256(p)) for p in sources}
    assert {(i["path"], i["sha256"]) for h in record["held_out"] for i in h["inputs"]} == {(p, sha256(p)) for p in held}
    assert all(i["size"] == Path(i["path"]).stat().st_size for i in record["inputs"])
    assert [r["image"] for r in record["images"]] == [s["image"] for s in config["sources"]]
    for row in record["images"]:
        assert row["kept"] == sum(1 for image in first["source_image"] if image == row["image"])
    bank = load_appearance_bank(BankReference(name=config["bank"], sha256=record["file"]["sha256"],
                                              directory=str(path.parent)))
    assert bank.digest == record["digest"] and bank.sha256 == record["file"]["sha256"]
    assert all(np.array_equal(bank.arrays[k], first[k]) for k in first)


# ---------------------------------------------------------------------------------------------------- GA3

@pytest.mark.contract
def test_load_refuses_a_changed_byte_a_wrong_reference_and_a_missing_file(file_banks, tmp_path, monkeypatch):
    root, config, (record, path), _ = file_banks("context-tissue2d-v1")
    changed = tmp_path / "changed"
    changed.mkdir()
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    (changed / path.name).write_bytes(bytes(data))
    reference = BankReference(name=config["bank"], sha256=record["file"]["sha256"], directory=str(changed))
    with pytest.raises(ValueError, match="SHA-256") as error:
        load_appearance_bank(reference)
    assert str(changed / path.name) in str(error.value) and record["file"]["sha256"] in str(error.value)
    wrong = BankReference(name=config["bank"], sha256="0" * 64, directory=str(path.parent))
    with pytest.raises(ValueError, match="0" * 64):
        load_appearance_bank(wrong)
    with pytest.raises(FileNotFoundError, match="nothing.npz"):
        load_appearance_bank(BankReference(name="nothing", sha256="0" * 64, directory=str(tmp_path)))
    monkeypatch.delenv("STARFINDER_APPEARANCE_BANKS", raising=False)
    with pytest.raises(ValueError, match="STARFINDER_APPEARANCE_BANKS"):
        load_appearance_bank(BankReference(name=config["bank"], sha256=record["file"]["sha256"]))
    monkeypatch.setenv("STARFINDER_APPEARANCE_BANKS", str(path.parent))
    assert load_appearance_bank(BankReference(name=config["bank"], sha256=record["file"]["sha256"])).digest == \
        record["digest"]


@pytest.mark.contract
def test_bank_reference_entries_and_build_refusals(tmp_path):
    for kwargs, field in ((dict(name="x"), "sha256"), (dict(name="a/b", sha256="0" * 64), "name"),
                          (dict(name="x", sha256="ABC"), "sha256"), (dict(preset="nothing"), "preset"),
                          (dict(preset="context-dense3d-v1", n=1), "n: a closed-form"),
                          (dict(preset="context-dense3d-v1", name="x"), "closed-form")):
        with pytest.raises(ValueError, match=field):
            BankReference(**kwargs)
    bank = cf_bank("context-tissue2d-v1")
    with pytest.raises(ValueError, match="no entry_id 'nothing/1'"):
        bank.entry("nothing/1")
    assert bank.entry(str(bank.arrays["entry_id"][0])) == 0
    case = bank_cases()["nuclei"]
    config = write_case(case, tmp_path / "case")
    with pytest.raises(ValueError, match="written as case-nuclei.npz"):
        build_appearance_bank(config, tmp_path / "other.npz")
    with pytest.raises(ValueError, match="unknown keys"):
        build_appearance_bank(dict(config, colour="red"), tmp_path / "case-nuclei.npz")
    held = dict(config, held_out=[dict(image="kept-aside", files=[config["sources"][0]["labels"]])])
    with pytest.raises(ValueError, match="held-out image kept-aside"):
        build_appearance_bank(held, tmp_path / "case-nuclei.npz")
    built(config, tmp_path)
    with pytest.raises(FileExistsError):
        build_appearance_bank(config, tmp_path / "case-nuclei.npz")


def _copy_with(config, root, change):
    """A copy of a build configuration whose first source's files are rewritten by ``change(kind, array)``."""
    root.mkdir(parents=True)
    copy = json.loads(json.dumps(config))
    source = copy["sources"][0]
    for key, spec in [*(("stain", (source["stains"], c)) for c in source["stains"]),
                      ("labels", (source, "labels")), ("labels", (source, "nucleus_labels"))]:
        holder, name = spec
        if name not in holder:
            continue
        array = tifffile.imread(holder[name])
        new = change(key, array)
        if new is not None:
            target = root / f"{key}-{name}.tif"
            tifffile.imwrite(target, new)
            holder[name] = str(target)
    return copy


@pytest.mark.contract
@pytest.mark.parametrize("preset", FILE_PRESETS)
def test_br4_refuses_both_copies_of_the_file_bank(file_banks, preset, tmp_path):
    root, config, _, _ = file_banks(preset)
    image = config["sources"][0]["image"]
    # Half a FOV in Y and X: the routine presets' FOVs are 64 voxels wide.
    half = context_scene_preset(preset)[1].sample.acquisition.fov_shape_zyx[1] // 2
    brighter = _copy_with(config, tmp_path / "brighter",
                          lambda kind, a: (a.astype(np.float32) * 2) if kind == "stain" else None)
    rolled = _copy_with(config, tmp_path / "rolled",
                        lambda kind, a: np.roll(a, (half, half), axis=(-2, -1)) if kind == "labels" else None)
    for copy, reason in ((brighter, "outside a factor 1.5 of the bank's median plateau"),
                         (rolled, "median over the visible stain")):
        loaded = []
        for source in copy["sources"]:
            stains = [tifffile.imread(source["stains"][c]).astype(np.float64) for c in copy["channels"]]
            labels = tifffile.imread(source.get("nucleus_labels") or source["labels"])
            if source.get("project_z"):
                stains, labels = [s.max(axis=0, keepdims=True) for s in stains], labels.max(axis=0, keepdims=True)
            loaded.append((oracle.plateau(labels, stains[0]), oracle.visible_median(stains[0])))
        refused, median = oracle.guard([p for p, _ in loaded], [v for _, v in loaded], 1.5)
        assert any(k == 0 for k, *_ in refused), (reason, refused, loaded)
        with pytest.raises(ValueError, match="BR4") as error:
            build_appearance_bank(copy, tmp_path / reason[:7].replace(" ", "-") / f"{copy['bank']}.npz")
        message = str(error.value)
        for k, kind, first, second in refused:
            assert f"source {copy['sources'][k]['image']}: its plateau {first:g}" in message
            assert f"{second:g}" in message
        assert f"source {image}" in message and reason in message


# ---------------------------------------------------------------------------------------------------- GA5

@pytest.mark.validation
@pytest.mark.parametrize("name", ["nuclei", "cells"])
def test_bank_cases_meet_their_known_outcomes(tmp_path, name):
    case = bank_cases()[name]
    record, path = built(write_case(case, tmp_path), tmp_path / "bank")
    bank = load_appearance_bank(BankReference(name=record["bank"], sha256=record["file"]["sha256"],
                                              directory=str(path.parent)))
    rows, counts, near = oracle.selection(case.labels.astype(np.int64), None if case.nucleus_labels is None
                                          else case.nucleus_labels.astype(np.int64), case.spacing,
                                          record["configuration"])
    assert {label: row["outcome"] for label, row in rows.items()} == case.expected
    assert near == []
    assert oracle.bank_entries(bank) == {(case.name, label) for label, o in case.expected.items() if o == "kept"}
    assert {k: record["images"][0][k] for k in counts} == counts
    assert record["images"][0]["plateau"] == case.plateau == bank.record["plateau_median"]
    if name == "cells":
        # The cell with two nucleus labels carries the larger (80 voxels) as its nucleus (rule 3 and EN1).
        assert rows[5]["nucleus_label"] == 5 and rows[5]["nucleus_voxels"] == 80
        assert int(bank.entries.set_index("source_label").loc[5, "nucleus_voxels"]) == 80


@pytest.mark.validation
@pytest.mark.parametrize("preset", ROUTINE)
def test_closed_form_bank_selection_equals_the_oracle(preset):
    bank = cf_bank(preset)
    configuration = bank.record["configuration"]
    kept, reported = set(), []
    for source, row in zip(cf_sources(preset), bank.record["images"]):
        assert row["image"] == source["image"]
        rows, counts, near = oracle.selection(source["labels"], source["nucleus_labels"], tuple(source["spacing"]),
                                              configuration)
        assert {k: row[k] for k in counts} == counts, source["image"]
        kept |= {(source["image"], label) for label, r in rows.items() if r["outcome"] == "kept"}
        reported += [(source["image"], label) for label in near]
    # Reported, not asserted: with a single candidate after rule 1, its size is both percentiles, so it lies on the
    # limit by construction; the integer sizes and the percentiles decide it with the build's arithmetic.
    print("GA5", preset, "labels within 1e-9 of a rule's limit:", reported)
    assert oracle.bank_entries(bank) == kept
    held = f"{preset}/seed-{HELD_OUT}"
    assert held not in set(bank.arrays["source_image"]) and held not in set(bank.arrays["images"])
    assert held not in set(bank.arrays["patch_image"])


# ---------------------------------------------------------------------------------------------------- GA6

def outside_bound(r, P):
    """GA6's outside-mask tolerance over the plateau: 1e-6 of the rim level, with the project's default absolute
    1e-9 as its floor where the rim level is 0 (the amendment of 2026-10-17 to GA6)."""
    return max(1e-6 * r / P, 1e-9)


def check_entries(bank, sources, report):
    """Every entry of ``bank`` against the oracle recomputed from its source arrays (GA6's tolerances)."""
    by_image = {s["image"]: s for s in sources}
    configuration = bank.record["configuration"]
    fall = configuration["fall_off"]
    reach = fall["length_um"] + 4 * fall["blur_um"]
    nuclei = {}
    for i, entry_id in enumerate(bank.arrays["entry_id"]):
        source = by_image[str(bank.arrays["source_image"][i])]
        label = int(bank.arrays["source_label"][i])
        P = float(bank.arrays["plateau"][i])
        if source["nucleus_labels"] is not None and source["image"] not in nuclei:
            nuclei[source["image"]] = oracle.selection(source["labels"], source["nucleus_labels"],
                                                       tuple(source["spacing"]), configuration)[0]
        nucleus_label = nuclei[source["image"]][label]["nucleus_label"] if source["image"] in nuclei else 0
        want = oracle.entry(source["stains"], source["labels"], source["nucleus_labels"], label, nucleus_label,
                            tuple(source["spacing"]), configuration)
        distance, excess, nucleus = oracle.stored(bank, str(entry_id))
        assert [s.start for s in want["box"]] == bank.arrays["box_origin"][i].tolist()
        assert list(distance.shape) == [s.stop - s.start for s in want["box"]]
        # D_e ≤ 0 exactly on the mask; D_e within 1e-6 µm of EDT(not M) − EDT(M).
        assert np.array_equal(distance <= 0, want["own"])
        assert np.max(np.abs(distance - want["distance"])) <= 1e-6
        assert excess.shape[0] == len(source["stains"]) == len(bank.channels)
        for c in range(len(source["stains"])):
            b, r, X, field = want["b"][c], want["r"][c], want["X"][c], want["field"][c]
            assert abs(bank.arrays["local_background"][i, c] - b / P) <= 1e-9 * max(1.0, b / P)
            assert abs(bank.arrays["rim_level"][i, c] - r / P) <= 1e-9 * max(1.0, r / P)
            inside = want["own"]
            # Inside: the source excess over the plateau, within one float32 rounding.
            assert np.all(np.abs(excess[c][inside] - X[inside] / P) <= oracle.FLOAT32 * np.abs(X[inside] / P))
            # Outside: OUT1's four steps, within GA6's bound; zero beyond λ + 1.0 µm.
            outside = ~inside
            error = np.abs(excess[c][outside] - field[outside] / P)
            assert np.all(error <= outside_bound(r, P)), (str(entry_id), c, float(error.max()), r / P)
            assert np.all(excess[c][want["d"] > reach] == 0)
            report.append((str(entry_id), c, float(np.max(error, initial=0)), r / P))
        voxel = np.prod(source["spacing"] if source["labels"].shape[0] > 1 else source["spacing"][1:])
        assert int(bank.arrays["voxels"][i]) == int(want["own"].sum())
        assert bank.arrays["volume"][i] == pytest.approx(int(want["own"].sum()) * voxel, abs=1e-9)
        if source["nucleus_labels"] is not None:
            assert nucleus is not None
            assert np.array_equal(nucleus <= 0, want["nucleus"])
            assert np.max(np.abs(nucleus - want["nucleus_distance"])) <= 1e-6
            assert int(bank.arrays["nucleus_voxels"][i]) == int(want["nucleus"].sum())


def foregrounds(bank, sources):
    """BG1's patch rule recomputed on each source image, by image ID."""
    settings = bank.record["configuration"]["patches"]
    plateaus = dict(zip(bank.arrays["images"], bank.arrays["image_plateau"]))
    return {s["image"]: oracle.foreground(s["labels"], s["nucleus_labels"], s["stains"], s["spacing"],
                                          plateaus[s["image"]], settings) for s in sources}


def check_patches(bank, sources):
    """Every patch's foreground share (BG1's patch rule, at most 10 %) recomputed at its origin."""
    masks = foregrounds(bank, sources)
    size = bank.arrays["patch"].shape[2:]
    limit = bank.record["configuration"]["patches"]["max_foreground"]
    for k, image in enumerate(bank.arrays["patch_image"]):
        share = oracle.patch_share(masks[str(image)], bank.arrays["patch_origin"][k], size)
        assert share == bank.arrays["patch_foreground"][k] and share <= limit


@pytest.mark.validation
@pytest.mark.parametrize("preset", ROUTINE)
def test_closed_form_entries_carry_what_the_page_states(preset):
    bank = cf_bank(preset)
    sources = cf_sources(preset)
    report = []
    check_entries(bank, sources, report)
    check_patches(bank, sources)
    assert len(bank.arrays["patch_image"]) > 0
    for row, source in zip(bank.record["images"], sources):
        reference = source["nucleus_labels"] if source["nucleus_labels"] is not None else source["labels"]
        assert row["plateau"] == oracle.plateau(reference, source["stains"][0])
        assert row["visible_median"] == oracle.visible_median(source["stains"][0])
    rims = {e: r for e, _, _, r in report}
    print("GA6", preset, "entries", len(bank), "largest difference outside over r",
          max((d / rims[e] for e, _, d, _ in report if rims[e] > 0), default=0.0))


def _source_of(case):
    return dict(image=case.name, stains=[np.asarray(s, dtype=np.float64) for s in case.stains.values()],
                labels=case.labels.astype(np.int64),
                nucleus_labels=None if case.nucleus_labels is None else case.nucleus_labels.astype(np.int64),
                spacing=np.asarray(case.spacing, dtype=np.float64))


@pytest.mark.validation
@pytest.mark.parametrize("name", ["nuclei", "cells"])
def test_bank_case_entries_and_patches(tmp_path, name):
    case = bank_cases()[name]
    record, path = built(write_case(case, tmp_path), tmp_path / "bank")
    bank = load_appearance_bank(BankReference(name=record["bank"], sha256=record["file"]["sha256"],
                                              directory=str(path.parent)))
    source = _source_of(case)
    check_entries(bank, [source], [])
    check_patches(bank, [source])
    assert len(bank.arrays["patch_image"]) > 0
    assert np.allclose(bank.arrays["local_background"][:, 0] * case.plateau, case.background, rtol=0, atol=1e-9)
    if name == "cells":
        # Cell 5: its nucleus is the 80-voxel label; the 60-voxel nucleus is not labelled, and its DAPI stays in the
        # excess (60 over the background 5, over the plateau 100); both channels are carried.
        entry = f"{case.name}/5"
        distance, excess, nucleus = oracle.stored(bank, entry)
        box = tuple(slice(o, o + n) for o, n in zip(bank.arrays["box_origin"][bank.entry(entry)],
                                                    bank.arrays["box_shape"][bank.entry(entry)]))
        assert np.array_equal(nucleus <= 0, case.nucleus_labels[box] == 5)
        other = case.nucleus_labels[box] == 6
        assert other.sum() == 60
        assert np.allclose(excess[0][other], (60.0 - 5.0) / 100.0, rtol=oracle.FLOAT32, atol=0)
        assert bank.channels == ("DAPI", "Flamingo") and excess.shape[0] == 2


def _fall_off_bank(tmp_path, name):
    case = fall_off_cases()[name]
    record, path = built(write_case(case, tmp_path / name), tmp_path / name / "bank")
    bank = load_appearance_bank(BankReference(name=record["bank"], sha256=record["file"]["sha256"],
                                              directory=str(path.parent)))
    return case, bank


@pytest.mark.validation
def test_fall_off_cases(tmp_path):
    banks = {name: _fall_off_bank(tmp_path, name) for name in ("a", "b10", "b20", "c")}
    case, bank = banks["a"]
    configuration = bank.record["configuration"]
    fall = configuration["fall_off"]
    entry = f"{case.name}/1"
    P, r = case.plateau, 10.0
    want = oracle.entry([np.asarray(case.stains["DAPI"])], case.labels.astype(np.int64), None, 1, 0, case.spacing,
                        configuration)
    assert want["b"] == [case.background] and want["r"] == [r]
    # (a) The cap binds everywhere and v = 1: outside, the 0.25-µm blur of r · max(0, 1 − d / λ), zero beyond λ + 1.
    d = want["d"]
    ramp = r * np.maximum(0.0, 1.0 - d / fall["length_um"])
    expected = ndi.gaussian_filter(ramp, oracle.sigmas(fall["blur_um"], case.spacing, ramp.shape))
    expected[d > fall["length_um"] + 4 * fall["blur_um"]] = 0.0
    _, excess, _ = oracle.stored(bank, entry)
    outside = ~want["own"]
    assert np.allclose(want["parts"][0]["v"][outside], 1.0, rtol=0, atol=1e-12)
    assert np.max(np.abs(excess[0][outside] - expected[outside] / P)) <= outside_bound(r, P)
    # (b) The neighbour at 10r and at 20r: the two copies of entry 1 are equal; the neighbour's stain never enters.
    first, second = (oracle.stored(banks[n][1], f"{n}/1") for n in ("b10", "b20"))
    assert all(np.array_equal(x, y) for x, y in zip(first[:2], second[:2]))
    neighbour = (banks["b10"][0].labels[tuple(slice(o, o + s) for o, s in zip(
        banks["b10"][1].arrays["box_origin"][banks["b10"][1].entry("b10/1")],
        banks["b10"][1].arrays["box_shape"][banks["b10"][1].entry("b10/1")]))] == 2)
    assert neighbour.any() and np.all(first[1][0][neighbour] <= r / P * (1 + 1e-6))
    # (c) The rising surround: the stored field equals OUT1 with the non-increasing step and differs without it.
    case_c, bank_c = banks["c"]
    want_c = oracle.entry([np.asarray(case_c.stains["DAPI"])], case_c.labels.astype(np.int64), None, 1, 0,
                          case_c.spacing, configuration)
    raw_c = oracle.entry([np.asarray(case_c.stains["DAPI"])], case_c.labels.astype(np.int64), None, 1, 0,
                         case_c.spacing, configuration, monotone=False)
    _, excess_c, _ = oracle.stored(bank_c, "c/1")
    out_c = ~want_c["own"]
    rc = want_c["r"][0]
    assert np.max(np.abs(excess_c[0][out_c] - want_c["field"][0][out_c] / P)) <= outside_bound(rc, P)
    gap = np.max(np.abs(raw_c["field"][0][out_c] - want_c["field"][0][out_c])) / rc
    print("GA6 fall_off_cases (c): largest difference without the non-increasing step, over r:", gap)
    assert gap > 1e-3
    # Every case against the oracle's four steps, and zero beyond λ + 1.0 µm.
    for name, (case_k, bank_k) in banks.items():
        check_entries(bank_k, [_source_of(case_k)], [])


# ---------------------------------------------------------------------------------------------------- GA14

@pytest.mark.validation
@pytest.mark.parametrize("preset", ROUTINE)
def test_brightness_is_relative_to_the_source_plateau(preset):
    bank = cf_bank(preset)
    sources = {s["image"]: s for s in cf_sources(preset)}
    plateaus = {str(i): float(p) for i, p in zip(bank.arrays["images"], bank.arrays["image_plateau"])}
    # O8's g: the median plateau of the bank's source images.
    assert bank.record["plateau_median"] == float(np.median(list(plateaus.values())))
    assert all(bank.arrays["plateau"][i] == plateaus[str(image)] for i, image in enumerate(bank.arrays["source_image"]))
    size = bank.arrays["patch"].shape[2:]
    masks = foregrounds(bank, list(sources.values()))
    levels = {}
    for k, image in enumerate(bank.arrays["patch_image"]):
        source = sources[str(image)]
        P = plateaus[str(image)]
        origin = bank.arrays["patch_origin"][k]
        box = tuple(slice(o, o + n) for o, n in zip(origin, size))
        holes = masks[str(image)][box]
        for c, stain in enumerate(source["stains"]):
            stored = bank.arrays["patch"][k, c].astype(np.float64)
            # Relative to the plateau: P · patch − source is one level shift per patch on the voxels not filled.
            shift = stored[~holes] * P - stain[box][~holes]
            assert np.ptp(shift) <= 4 * oracle.FLOAT32 * P * np.abs(stored).max(), (image, k, c)
            levels.setdefault((str(image), int(origin[0]), c), []).append(stored.mean())
    for key, means in levels.items():
        assert np.ptp(means) <= 1e-6 * max(1.0, abs(np.mean(means))), key


# ---------------------------------------------------------------------------------------------------- GA22

_AUDIT = r"""
import json, os, sys
from pathlib import Path
import starfinder.synthetic
from starfinder.synthetic import closed_form_bank
import scipy.ndimage, skimage.filters, tifffile, pandas
opened = []
def audit(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
        opened.append(os.fsdecode(arguments[0]))
sys.addaudithook(audit)
bank = closed_form_bank(sys.argv[1], 4)
sys.stdout.write(json.dumps(dict(opened=opened, inputs=bank.record["inputs"], held_out=bank.record["held_out"],
                                 entries=len(bank))))
"""


@pytest.mark.contract
@pytest.mark.parametrize("preset", ROUTINE)
def test_closed_form_bank_reads_no_file_outside_the_package(preset, tmp_path):
    import starfinder
    env = dict(os.environ, TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", _AUDIT, preset], cwd=tmp_path, env=env, capture_output=True,
                            text=True, check=True)
    record = json.loads(result.stdout)
    allowed = [Path(tmp_path).resolve(), Path(starfinder.__file__).resolve().parent,
               *{Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}]
    outside = sorted({p for p in record["opened"]
                      if not any(Path(p).resolve().is_relative_to(root) for root in allowed)})
    print("GA22", preset, "files opened during the build:", len(record["opened"]), "outside:", outside)
    assert outside == []
    assert [(i["preset"], i["seed"]) for i in record["inputs"]] == [(preset, s) for s in SEEDS if s != HELD_OUT]
    assert [i["seed"] for h in record["held_out"] for i in h["inputs"]] == [HELD_OUT]
    assert not any(key in i for i in record["inputs"] for key in ("path", "sha256"))
