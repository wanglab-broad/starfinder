"""Sequencing and other rounds handled separately, the morphology entry and its saved form (W-337).

docs/coordination.md ("Sequencing rounds and other rounds"), docs/registration-contract.md
("Other rounds and the reference stain") and docs/checkpoints.md ("Prepared morphology
images"). The dataset is written in session as single-channel TIFFs (8×32×32, uint16):
the reference round1 (sequencing ch00–ch03 and the stain file ch04, DAPI), the sequencing
round2 (round1's sequencing channels displaced by (0, 1, −1)) and the other round morph
(Flamingo, RBD, DAPI). The textures are those of test_registration_other_rounds: the stain
is texture 0 in ch04 of round1 and in morph's DAPI, and morph is the truth displaced by
d = (0, 3, −2), so its pull map is p + d.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pytest
from skimage.measure import label

from starfinder.barcode import Codebook, NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.dataset import (ChannelInfo, CheckpointConfig, Dataset, ExecutionConfig, ExternalReference,
                                MorphologyConfig, PipelineConfig, RegistrationRecipe, RegistrationStep, RoundState)
from starfinder.dataset._prepared import SAME_ACQUISITION
from starfinder.dataset.fov import _rotated
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.io._checkpoint import FORMAT_VERSION, STAGES
from starfinder.registration import (RegistrationSignalConfig, TranslationConfig, WarpConfig, apply_transform)
from starfinder.segmentation import (SEGMENTATION_METHODS, FlamingoEnhancementConfig, InputChannel, SegmentationPlan,
                                     SegmentationRun, SegmentationSpec, enhance_with_flamingo)
from starfinder.segmentation._labels import array_sha256
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingPlan

from .test_registration_other_rounds import SHAPE, SHIFT, grid, texture

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

SEQUENCING = tuple(ChannelInfo(c, "seq") for c in ("ch00", "ch01", "ch02", "ch03"))
LABELS = tuple(c.channel for c in SEQUENCING)
STAIN = ChannelInfo("ch04", "DAPI", 405)
MORPH = (ChannelInfo("ch00", "Flamingo", 488), ChannelInfo("ch01", "RBD", 561), ChannelInfo("ch02", "DAPI"))
SEQUENCING_SHIFT = (0, 1, -1)
# The valid overlap of the pull map p + (0, 3, -2): y + 3 <= 31 and x - 2 >= 0.
OVERLAP = (slice(None), slice(0, 29), slice(2, 32))
REGISTRATION = RegistrationRecipe((RegistrationStep(TranslationConfig()),))


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """SimpleITK at one thread, as the project contract requires."""
    sitk = pytest.importorskip("SimpleITK")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """The input root and the written arrays: round1 (ZYXC), its stain (ZYX), round2, morph and morph's truth."""
    root = tmp_path_factory.mktemp("w337")
    p = grid()
    round1 = np.stack([texture(s, p) for s in (1, 2, 3, 6)], axis=-1)
    stain = texture(0, p)
    round2 = np.ascontiguousarray(np.roll(round1, SEQUENCING_SHIFT, axis=(0, 1, 2)))
    morph = np.stack([texture(s, p - SHIFT.reshape(3, 1, 1, 1)) for s in (4, 5, 0)], axis=-1)
    truth = np.stack([texture(s, p) for s in (4, 5, 0)], axis=-1)
    files = {"round1": [*zip(LABELS, np.moveaxis(round1, -1, 0)), ("ch04", stain)],
             "round2": list(zip(LABELS, np.moveaxis(round2, -1, 0))),
             "morph": list(zip((c.channel for c in MORPH), np.moveaxis(morph, -1, 0)))}
    for name, channels in files.items():
        for channel, image in channels:
            save_volume(np.ascontiguousarray(image), root / "in" / name / "FOV" / f"FOV_{channel}.tif",
                        metadata=ImageMetadata(f"FOV/{name}"))
    return dict(root=root, round1=round1, stain=stain, round2=round2, morph=morph, truth=truth)


def make_dataset(raw, *, other=True, stains=(STAIN,), morph=MORPH, rounds=None, output="out"):
    """round1 (reference) and round2 sequencing; with other, the other round morph; the reference stains."""
    root = raw["root"]
    rounds = rounds or RoundState(["round1", "round2"], ["morph"] if other else [], "round1")
    ds = Dataset(root / "in", root / output, "data", "sample", "out", rounds, SEQUENCING,
                 other_channel_order={"morph": morph} if "morph" in rounds.other_rounds else {},
                 reference_stains=stains)
    ds.codebook = Codebook(pd.DataFrame({"gene_id": [f"gene{i}" for i in range(6)],
                                         "color_sequence": ["11", "12", "21", "22", "33", "44"]}),
                           ("round1", "round2"), ds.channel_order)
    return ds


def pipeline(rotation=None, **change):
    return PipelineConfig(load=ImageLoadConfig(channel_labels=LABELS), rotation_degrees=rotation,
                          registration=REGISTRATION, spot_finding=LocalMaximaConfig("adaptive", .1),
                          extraction=NeighborhoodSumConfig((0, 1, 1)), decoding=WtaDecoderConfig(),
                          filtering=ReadFilterConfig(), **change)


def stain_reference(fov):
    """The ExternalReference the entry registers on: the reference stain, labelled as the nuclei_registration rule."""
    return ExternalReference(np.ascontiguousarray(fov.images["reference_stain"][..., 0]),
                             fov.metadata["reference_stain"], "round1:ch04")


def stain_recipe():
    return RegistrationRecipe((RegistrationStep(TranslationConfig()),),
                              signal=RegistrationSignalConfig("channel", reference_channel="ch04", moving_channel="ch02"))


def files(directory):
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


def masked(path):
    """A file's bytes with the OME UUID masked: tifffile writes a new random UUID into every OME-TIFF."""
    data = path.read_bytes()
    return re.sub(rb"urn:uuid:[0-9a-f-]{36}", b"urn:uuid:" + b"0" * 36, data) if path.name.endswith(".ome.tif") else data


def without(data, *keys):
    return {k: v for k, v in data.items() if k not in keys}


# --- FOV.run leaves other rounds alone --------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_run_neither_loads_nor_changes_an_other_round(raw, mode, tmp_path):
    execution = ExecutionConfig(mode)
    runs = {}
    for case, other, stains in (("with", True, (STAIN,)), ("without", False, ())):
        checkpoints = CheckpointConfig(directory=tmp_path / case)
        fov = make_dataset(raw, other=other, stains=stains).fov("FOV").run(pipeline(), execution=execution,
                                                                           checkpoints=checkpoints)
        runs[case] = fov, tmp_path / case / "FOV"
    (fov, directory), (plain, plain_directory) = runs["with"], runs["without"]
    resident = {"batch": {"round1", "round2"}, "streaming": {"round1"}}[mode]
    assert set(fov.images) == resident == set(plain.images) and set(fov.metadata) == {"round1", "round2"}
    assert not {"morph", "reference_stain"} & (set(fov.load_diagnostics) | set(fov.registration_results))
    assert "rounds" not in fov.registration_record
    # The files written are the same, and the registered checkpoint lists the reference and sequencing rounds only.
    assert files(directory) == files(plain_directory)
    assert not any(name.startswith("other_rounds") for name in files(directory))
    assert sorted(name for name in files(directory) if name.endswith(".tif")) == [
        "registered/round1.ome.tif", "registered/round2.ome.tif"]
    header = json.loads((directory / "registered" / "transforms.json").read_text())
    assert header["image_rounds"] == ["round1", "round2"] and list(header["transforms"]) == ["round2"]
    assert list(header["registration_attempts"]) == ["round2"] and list(header["channels"]) == ["round1", "round2"]
    # Every sequencing output is byte-identical to the run of the dataset without the other round and the stain
    # (an OME-TIFF apart from its random UUID, which differs between any two writes).
    for name in ("registered/round1.ome.tif", "registered/round2.ome.tif", "candidates.csv", "pre_qc.csv"):
        assert masked(directory / name) == masked(plain_directory / name), name
    for name in ("registered/transforms.json", "candidates.json", "pre_qc.json"):
        written, expected = (json.loads((d / name).read_text()) for d in (directory, plain_directory))
        assert written["rounds"]["other_rounds"] == ["morph"] and expected["rounds"]["other_rounds"] == []
        assert without(written, "rounds") == without(expected, "rounds"), name
    assert fov.metadata == plain.metadata
    for name in resident:
        assert np.array_equal(fov.images[name], plain.images[name])
    pd.testing.assert_frame_equal(fov.spot_result.spots, plain.spot_result.spots)
    assert np.array_equal(fov.intensity_result.values, plain.intensity_result.values)
    pd.testing.assert_frame_equal(fov.decoding_result.table, plain.decoding_result.table)
    pd.testing.assert_frame_equal(fov.filtering_result.table, plain.filtering_result.table)
    # round2(q) = round1(q - s), so the pull map is p + s.
    assert len(fov.spot_result.spots) > 0 and fov.registration_results["round2"][0].transform.displacement_zyx == \
        tuple(map(float, SEQUENCING_SHIFT))


def test_run_keeps_a_resident_other_round_unchanged(raw):
    fov = make_dataset(raw).fov("FOV").load_images(rounds=["morph"])
    before = fov.images["morph"].copy(), fov.metadata["morph"]
    fov.run(pipeline(rotation=90))
    assert np.array_equal(fov.images["morph"], before[0]) and fov.metadata["morph"] == before[1]
    assert "rotation" not in fov.load_diagnostics["morph"] and list(fov.registration_results) == ["round2"]


def test_a_reference_round_that_is_an_other_round_is_processed_as_the_reference(raw):
    rounds = RoundState(["round2"], ["round1", "morph"], "round1")
    fov = make_dataset(raw, rounds=rounds).fov("FOV").run(PipelineConfig(load=ImageLoadConfig(channel_labels=LABELS),
                                                                         registration=REGISTRATION))
    assert set(fov.images) == {"round1", "round2"} and list(fov.registration_results) == ["round2"]
    assert np.array_equal(fov.images["round1"], raw["round1"])


def test_run_refuses_a_detection_plan_with_an_other_round(raw):
    fov = make_dataset(raw).fov("FOV")
    config = replace(pipeline(), spot_finding=SpotFindingPlan(LocalMaximaConfig("adaptive", .1),
                                                              rounds=("round1", "morph")),
                     extraction=None, decoding=None, filtering=None)
    with pytest.raises(ValueError, match=r"detection rounds \['morph'\] are not processed by FOV.run"):
        fov.run(config)
    assert fov.images == {}


# --- FOV.register ----------------------------------------------------------------------------------------

def test_register_defaults_to_the_sequencing_rounds_and_refuses_an_other_round(raw):
    fov = make_dataset(raw).fov("FOV").load_images(rounds=["round1", "round2"]).load_images(rounds=["morph"])
    morph = fov.images["morph"].copy()
    with pytest.raises(ValueError, match=r"rounds \['morph'\] are other rounds.*FOV\.register_rounds"):
        fov.register(REGISTRATION, rounds=["morph"])
    assert fov.registration_results == {} and fov.registration_attempts == {}
    fov.register(REGISTRATION)
    assert list(fov.registration_results) == list(fov.registration_chains) == ["round2"]
    assert list(fov.registration_attempts) == ["round2"]
    assert np.array_equal(fov.images["morph"], morph) and fov.metadata["morph"] == ImageMetadata("FOV/morph")
    with pytest.raises(ValueError, match="FOV.register_rounds"):
        fov.register(REGISTRATION, rounds=["round2", "morph"])


# --- Rotation --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [90, 30])
def test_load_images_rotates_as_run_does_and_a_round_is_rotated_once(raw, angle):
    ds = make_dataset(raw)
    loaded = ds.fov("FOV").load_images(rounds=["round1", "round2"], rotation_degrees=angle)
    run = ds.fov("FOV").run(PipelineConfig(load=ImageLoadConfig(channel_labels=LABELS), rotation_degrees=angle))
    for name in ("round1", "round2"):
        assert np.array_equal(loaded.images[name], run.images[name])
        assert loaded.images[name].dtype == np.uint16
        assert loaded.metadata[name] == run.metadata[name] == ImageMetadata(f"FOV/{name}").rotated(
            SHAPE, angle, frame_id=f"FOV/{name}/rotate:{angle}")
        assert loaded.load_diagnostics[name]["rotation"] == run.load_diagnostics[name]["rotation"]
        assert loaded.load_diagnostics[name]["rotation"]["angle_degrees"] == angle
    expected = _rotated(raw["round1"], ImageMetadata("FOV/round1"), angle)[0]
    assert np.array_equal(loaded.images["round1"], expected)
    if angle == 90:
        assert np.array_equal(loaded.images["round1"], np.rot90(raw["round1"], k=1, axes=(1, 2)))
    # A second rotation raises, by every route, and changes nothing.
    image = loaded.images["round1"].copy()
    with pytest.raises(ValueError, match="already rotated by .* a round is rotated once"):
        loaded.rotate(angle=angle)
    with pytest.raises(ValueError, match="a round is rotated once"):
        run.run(PipelineConfig(rotation_degrees=angle))
    with pytest.raises(ValueError, match="a round is rotated once"):
        loaded._rotate_round(round_name="round2", angle=angle)
    assert np.array_equal(loaded.images["round1"], image)
    # A new load replaces the round, so it can be rotated again.
    loaded.load_images(rounds=["round1"], rotation_degrees=angle)
    assert np.array_equal(loaded.images["round1"], image)


def test_load_images_refuses_an_invalid_angle_and_an_unknown_stain(raw):
    with pytest.raises(ValueError, match="rotation_degrees must be finite"):
        make_dataset(raw).fov("FOV").load_images(rounds=["round1"], rotation_degrees=float("nan"))
    with pytest.raises(ValueError, match="reference_stain when the dataset has reference stains"):
        make_dataset(raw, stains=()).fov("FOV").load_images(rounds=["reference_stain"])


# --- The reference stain ---------------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [None, 90, 30])
def test_the_reference_stain_is_the_stain_file_rotated_as_the_reference_round(raw, angle):
    fov = make_dataset(raw).fov("FOV").run(PipelineConfig(load=ImageLoadConfig(channel_labels=LABELS),
                                                          rotation_degrees=angle, registration=REGISTRATION))
    fov.prepare_morphology(MorphologyConfig(rotation_degrees=angle))
    stain, metadata = fov.images["reference_stain"], fov.metadata["reference_stain"]
    assert stain.shape == SHAPE + (1,) and stain.dtype == np.uint16
    if angle is None:
        expected = raw["stain"]
    else:
        expected = _rotated(raw["stain"], ImageMetadata("FOV/round1"), angle)[0]
        # The reference round's channels were rotated by the same code.
        assert np.array_equal(fov.images["round1"][..., 0],
                              _rotated(raw["round1"][..., 0], ImageMetadata("FOV/round1"), angle)[0])
    if angle == 90:
        assert np.array_equal(expected, np.rot90(raw["stain"], k=1, axes=(1, 2)))
    assert np.array_equal(stain[..., 0], expected)
    assert metadata == fov.metadata["round1"]
    assert fov.registration_record["rounds"]["reference_stain"] == dict(
        relation=SAME_ACQUISITION, reference="round1", reference_sha256=None, recipe=None)
    assert SAME_ACQUISITION == "same acquisition as the reference round"
    for state in (fov.registration_results, fov.registration_chains, fov.registration_attempts,
                  fov.registration_record["application"]):
        assert "reference_stain" not in state
    assert list(fov.registration_attempts) == ["round2", "morph"]


# --- The other round -------------------------------------------------------------------------------------

def test_the_entry_recovers_the_known_translation_and_equals_the_step_by_step_calls(raw):
    ds = make_dataset(raw)
    fov = ds.fov("FOV").prepare_morphology()            # the raw files only: no FOV.run
    chain = fov.registration_chains["morph"]
    assert [t.displacement_zyx for t in chain.transforms] == [(0.0, 3.0, -2.0)]
    assert fov.registration_results["morph"][0].transform.displacement_zyx == tuple(SHIFT)
    registered = fov.images["morph"]
    assert registered.dtype == np.uint16 and registered.shape == SHAPE + (3,)
    once = apply_transform(raw["morph"], chain, config=WarpConfig())
    for c in range(3):
        # Every channel is resampled once through the chain, and equals the truth on the valid overlap.
        assert np.array_equal(registered[..., c], once[..., c])
        assert np.array_equal(registered[OVERLAP + (c,)], raw["truth"][OVERLAP + (c,)])
    assert fov.metadata["morph"] == fov.metadata["reference_stain"] == ImageMetadata("FOV/round1")
    digest = hashlib.sha256(np.ascontiguousarray(raw["stain"]).tobytes()).hexdigest()
    entry = fov.registration_record["rounds"]["morph"]
    assert entry["reference"] == "round1:ch04" and entry["reference_sha256"] == digest
    assert entry["recipe"]["steps"] == ["translation"]
    assert entry["recipe"]["signal"] == dict(mode="channel", reference_channel="ch04", moving_channel="ch02")
    assert [a["record"] for a in fov.registration_attempts["morph"]] == ["estimation", "application"]

    steps = ds.fov("FOV").load_images(rounds=["reference_stain"]).load_images(rounds=["morph"])
    steps.register_rounds(stain_recipe(), rounds=["morph"], reference=stain_reference(steps))
    for name in ("reference_stain", "morph"):
        assert np.array_equal(steps.images[name], fov.images[name]) and steps.metadata[name] == fov.metadata[name]
    assert steps.registration_chains["morph"].transforms == chain.transforms
    assert steps.registration_results == fov.registration_results
    assert steps.registration_attempts == fov.registration_attempts
    assert steps.registration_record["rounds"]["morph"] == entry
    assert without(fov.registration_record, "rounds") == without(steps.registration_record, "rounds")
    # The entry adds the reference stain's statement; the step-by-step calls make no such entry.
    assert set(fov.registration_record["rounds"]) - set(steps.registration_record["rounds"]) == {"reference_stain"}


def test_a_given_recipe_on_the_stain_and_one_on_the_reference_round(raw):
    by_name = RegistrationRecipe((RegistrationStep(TranslationConfig()),),
                                 signal=RegistrationSignalConfig("channel", reference_channel="DAPI", moving_channel="DAPI"))
    fov = make_dataset(raw).fov("FOV").prepare_morphology(MorphologyConfig(recipe=by_name))
    assert fov.registration_chains["morph"].transforms[0].displacement_zyx == (0.0, 3.0, -2.0)
    assert fov.registration_record["rounds"]["morph"]["reference"] == "round1:ch04"
    on_round = RegistrationRecipe((RegistrationStep(TranslationConfig()),),
                                  signal=RegistrationSignalConfig("channel", reference_channel=0, moving_channel=0))
    with pytest.raises(ValueError, match="reference round 'round1', which is not resident"):
        make_dataset(raw).fov("FOV").prepare_morphology(MorphologyConfig(recipe=on_round))
    fov = make_dataset(raw).fov("FOV").load_images(rounds=["round1"]).prepare_morphology(MorphologyConfig(recipe=on_round))
    assert fov.registration_record["rounds"]["morph"]["reference"] == "round1"
    assert all(a.get("reference", "round1") == "round1" for a in fov.registration_attempts["morph"])
    mixed = RegistrationRecipe((RegistrationStep(TranslationConfig(), signal=RegistrationSignalConfig("max")),),
                               signal=RegistrationSignalConfig("channel", reference_channel="ch04", moving_channel="ch02"))
    with pytest.raises(ValueError, match="names a reference stain in some signals only"):
        make_dataset(raw).fov("FOV").prepare_morphology(MorphologyConfig(recipe=mixed))


@pytest.mark.parametrize("stains, morph, message", [
    ((), MORPH, "the dataset has no reference stain"),
    ((STAIN, ChannelInfo("ch05", "PI")), MORPH, "the dataset has 2 reference stains"),
    ((STAIN,), MORPH[:2] + (ChannelInfo("ch02", "Nissl"),), "the round has 0 such channels"),
    ((STAIN,), MORPH[:2] + (ChannelInfo("ch02", "DAPI"), ChannelInfo("ch03", "DAPI")), "the round has 2 such channels"),
    ((ChannelInfo("ch04"),), MORPH, "the round has 0 such channels"),
])
def test_without_a_recipe_the_shared_stain_must_be_single(raw, stains, morph, message):
    fov = make_dataset(raw, stains=stains, morph=morph).fov("FOV")
    with pytest.raises(ValueError, match=message) as error:
        fov.prepare_morphology()
    assert "MorphologyConfig(recipe=...)" in str(error.value)
    assert fov.images == {} and fov.registration_record == {}


def test_morphology_config_and_entry_arguments_are_checked(raw):
    assert MorphologyConfig(rounds=["morph"]).rounds == ("morph",)
    with pytest.raises(ValueError, match="finite"):
        MorphologyConfig(rotation_degrees=float("inf"))
    with pytest.raises(TypeError):
        MorphologyConfig(recipe=TranslationConfig())
    with pytest.raises(ValueError):
        MorphologyConfig(rounds=("morph", "morph"))
    with pytest.raises(TypeError):
        MorphologyConfig(rounds="morph")
    fov = make_dataset(raw).fov("FOV")
    with pytest.raises(ValueError, match=r"\['round2'\] are not other rounds"):
        fov.prepare_morphology(MorphologyConfig(rounds=("round2",)))
    with pytest.raises(TypeError):
        fov.prepare_morphology(PipelineConfig())
    with pytest.raises(ValueError, match="nothing to prepare"):
        make_dataset(raw, other=False, stains=()).fov("FOV").prepare_morphology()
    fov.prepare_morphology()
    with pytest.raises(ValueError, match="already prepared"):
        fov.prepare_morphology()


# --- Saved form ------------------------------------------------------------------------------------------

def test_the_saved_form_reloads_equal_values_and_checks_hashes_folders_and_identity(raw, tmp_path):
    ds = make_dataset(raw)
    checkpoints = CheckpointConfig(directory=tmp_path / "ck")
    fov = ds.fov("FOV").prepare_morphology(checkpoints=checkpoints)
    directory = tmp_path / "ck" / "FOV"
    assert files(directory) == ["other_rounds/morph/image.ome.tif", "other_rounds/morph/registration.json",
                                "other_rounds/reference_stain/image.ome.tif",
                                "other_rounds/reference_stain/registration.json"]
    record = json.loads((directory / "other_rounds" / "morph" / "registration.json").read_text())
    assert (record["dataset_id"], record["sample_id"], record["fov_id"], record["subtile_id"]) == (
        "data", "sample", "FOV", None)
    assert record["channels"] == [c.record() for c in MORPH] and record["rotation"] is None
    assert record["registration"]["reference"] == "round1:ch04" and len(record["registration"]["attempts"]) == 2
    assert record["image"]["sha256"] == hashlib.sha256(fov.images["morph"].tobytes()).hexdigest()
    assert {"code", "environment"} <= set(record["software"])
    stain_record = json.loads((directory / "other_rounds" / "reference_stain" / "registration.json").read_text())
    assert stain_record["registration"]["relation"] == SAME_ACQUISITION
    assert stain_record["registration"]["transforms"] is None and stain_record["registration"]["attempts"] is None

    fresh = ds.fov("FOV")
    for name in ("reference_stain", "morph"):
        fresh.load_registered_round(name, checkpoints=checkpoints)
        assert np.array_equal(fresh.images[name], fov.images[name]) and fresh.images[name].dtype == np.uint16
        assert fresh.metadata[name] == fov.metadata[name]
        entry = fresh.registration_record["rounds"][name]
        assert without(entry, "saved") == fov.registration_record["rounds"][name]
        path = f"other_rounds/{name}/image.ome.tif"
        assert entry["saved"] == {"path": path, "sha256": hashlib.sha256((directory / path).read_bytes()).hexdigest()}
    assert fresh.registration_chains["morph"].transforms == fov.registration_chains["morph"].transforms
    assert list(fresh.registration_chains) == ["morph"]
    assert fresh.registration_results == fov.registration_results
    assert fresh.registration_attempts == fov.registration_attempts
    assert fresh.registration_record["application"] == fov.registration_record["application"]

    # An existing folder raises before anything is loaded, unless overwrite=True.
    again = ds.fov("FOV")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        again.prepare_morphology(checkpoints=checkpoints)
    assert again.images == {}
    with pytest.raises(FileExistsError):
        again.load_images(rounds=["morph"]).register_rounds(stain_recipe(), rounds=["morph"], reference=ExternalReference(
            raw["stain"], ImageMetadata("FOV/round1"), "round1:ch04"), checkpoints=checkpoints)
    assert again.registration_attempts == {}
    written = files(directory)
    ds.fov("FOV").prepare_morphology(checkpoints=replace(checkpoints, overwrite=True))
    assert files(directory) == written
    ds.fov("FOV").load_registered_round("morph", checkpoints=checkpoints)

    # Another FOV's folder raises on the identity.
    shutil.copytree(directory / "other_rounds", tmp_path / "ck" / "FOV2" / "other_rounds")
    with pytest.raises(ValueError, match="has fov_id 'FOV', this FOV 'FOV2'"):
        ds.fov("FOV2").load_registered_round("morph", checkpoints=checkpoints)
    # A changed image file raises naming both hashes.
    image_path = directory / "other_rounds" / "morph" / "image.ome.tif"
    recorded = json.loads((image_path.parent / "registration.json").read_text())["image"]["file_sha256"]
    assert recorded == hashlib.sha256(image_path.read_bytes()).hexdigest()
    save_volume(fov.images["morph"] + 1, image_path, metadata=fov.metadata["morph"])
    changed = hashlib.sha256(image_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError) as error:
        ds.fov("FOV").load_registered_round("morph", checkpoints=checkpoints)
    assert recorded in str(error.value) and changed in str(error.value) and recorded != changed
    # A resident reference round on another grid raises.
    rotated = ds.fov("FOV").load_images(rounds=["round1"], rotation_degrees=90)
    with pytest.raises(ValueError, match="unlike the reference round"):
        rotated.load_registered_round("reference_stain", checkpoints=checkpoints)


def test_register_rounds_writes_the_same_saved_form(raw, tmp_path):
    ds = make_dataset(raw)
    checkpoints = CheckpointConfig(directory=tmp_path / "ck")
    fov = ds.fov("FOV").load_images(rounds=["reference_stain"]).load_images(rounds=["morph"])
    fov.register_rounds(stain_recipe(), rounds=["morph"], reference=stain_reference(fov), checkpoints=checkpoints)
    assert files(tmp_path / "ck" / "FOV") == ["other_rounds/morph/image.ome.tif",
                                              "other_rounds/morph/registration.json"]
    entry = ds.fov("FOV").prepare_morphology()
    loaded = ds.fov("FOV").load_registered_round("morph", checkpoints=checkpoints)
    assert np.array_equal(loaded.images["morph"], entry.images["morph"])
    assert loaded.registration_chains["morph"].transforms == entry.registration_chains["morph"].transforms


def test_run_resumes_from_candidates_beside_a_reloaded_morphology_image(raw, tmp_path):
    ds = make_dataset(raw)
    checkpoints = CheckpointConfig(directory=tmp_path / "ck")
    first = ds.fov("FOV").run(pipeline(), checkpoints=checkpoints).prepare_morphology(checkpoints=checkpoints)
    later = ds.fov("FOV").load_checkpoint("candidates", checkpoints=checkpoints)
    later.load_registered_round("morph", checkpoints=checkpoints)
    later.run(PipelineConfig(decoding=WtaDecoderConfig(), filtering=ReadFilterConfig()))
    pd.testing.assert_frame_equal(later.filtering_result.table, first.filtering_result.table)
    assert set(later.images) == {"morph"} and np.array_equal(later.images["morph"], first.images["morph"])


# --- The registered checkpoint ---------------------------------------------------------------------------

def test_the_registered_header_records_the_channels_and_reloads_without_them(raw, tmp_path):
    ds = make_dataset(raw)
    checkpoints = CheckpointConfig(directory=tmp_path / "ck", stages=("registered",))
    fov = ds.fov("FOV").run(PipelineConfig(load=ImageLoadConfig(channel_labels=LABELS), registration=REGISTRATION),
                            checkpoints=checkpoints)
    path = tmp_path / "ck" / "FOV" / "registered" / "transforms.json"
    header = json.loads(path.read_text())
    assert header["format_version"] == FORMAT_VERSION == 2
    assert header["channels"] == {name: [c.record() for c in SEQUENCING] for name in ("round1", "round2")}
    assert header["channel_labels"] == list(LABELS)
    assert STAGES == ("registered", "candidates", "pre_qc") and CheckpointConfig().stages == STAGES
    restored = []
    for keep in (True, False):
        if not keep:
            path.write_text(json.dumps(without(header, "channels")))
        restored.append(ds.fov("FOV").load_checkpoint("registered", checkpoints=checkpoints))
    for loaded in restored:
        assert set(loaded.images) == {"round1", "round2"}
        for name in loaded.images:
            assert np.array_equal(loaded.images[name], fov.images[name]) and loaded.metadata[name] == fov.metadata[name]
        assert loaded.registration_chains["round2"].transforms == fov.registration_chains["round2"].transforms
        assert loaded.registration_attempts == restored[0].registration_attempts
        assert loaded.registration_record == restored[0].registration_record
    assert restored[0].registration_record["application"] == fov.registration_record["application"]


def test_the_registered_checkpoint_holds_no_other_round(raw, tmp_path):
    fov = make_dataset(raw).fov("FOV").load_images(rounds=["round1", "round2"]).prepare_morphology()
    fov.save_checkpoint("registered", checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    header = json.loads((tmp_path / "ck" / "FOV" / "registered" / "transforms.json").read_text())
    assert header["image_rounds"] == ["round1", "round2"] and header["transforms"] == {}
    assert header["registration_attempts"] == {} and header["applications"] == {}
    assert files(tmp_path / "ck" / "FOV") == ["registered/round1.ome.tif", "registered/round2.ome.tif",
                                              "registered/transforms.json"]


# --- Segmentation in one process -------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdConfig:
    """A test-only method: the connected components of channel 0 above level."""
    level: float = 0.0
    method: str = field(default="w337_threshold", init=False)


def _threshold(image, config, context):
    labels = label(image[..., 0] > config.level, connectivity=1).astype(np.int32)
    return labels, {"effective": {"level": config.level}}


THRESHOLD = SegmentationSpec("w337_threshold", _threshold, targets=frozenset({"nucleus"}),
                             roles=frozenset({"nuclear"}), required_roles=(frozenset({"nuclear"}),),
                             seeds="optional", dimensions=frozenset({2, 3}), models=False, devices=frozenset({"cpu"}))


@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setitem(SEGMENTATION_METHODS, ThresholdConfig, THRESHOLD)


def plan(channel, level):
    return SegmentationPlan((SegmentationRun("nucleus", "nucleus", (channel,), ThresholdConfig(level)),))


STAIN_INPUT = InputChannel("nuclear", round="reference_stain", channel="DAPI")
ENHANCED_INPUT = InputChannel("nuclear", round="morph", channel="DAPI", prepare=FlamingoEnhancementConfig(),
                              prepare_channel="Flamingo")


def test_segment_reads_the_prepared_images_in_one_process_and_after_a_reload(raw, threshold, tmp_path):
    ds = make_dataset(raw)
    checkpoints = CheckpointConfig(directory=tmp_path / "ck")
    fov = ds.fov("FOV").run(pipeline(rotation=90)).prepare_morphology(MorphologyConfig(rotation_degrees=90),
                                                                      checkpoints=checkpoints)
    stain = fov.images["reference_stain"][..., 0]
    enhanced, _ = enhance_with_flamingo(fov.images["morph"][..., 2], fov.images["morph"][..., 0])
    level = float(np.mean(stain))
    results = {}
    for name, channel, expected in (("stain", STAIN_INPUT, stain), ("enhanced", ENHANCED_INPUT, enhanced)):
        threshold_level = float(np.mean(expected))
        fov.segment(plan(channel, threshold_level))
        result = fov.segmentation_results["nucleus"]
        results[name] = result
        (entry,) = result.record["input"]["channels"]
        # The input is the prepared channel, or the enhancement of the two prepared channels.
        assert entry["sha256"] == array_sha256(expected)
        assert np.array_equal(result.labels, label(expected > threshold_level, connectivity=1))
        assert result.labels.max() > 0
        assert (entry["round"], entry["channel"], entry["name"]) == (channel.round, "DAPI", "DAPI")
    stain_entry = results["stain"].record["input"]["channels"][0]
    assert stain_entry["wavelength"] == 405.0
    assert stain_entry["registration"] == {"reference": "round1", "reference_sha256": None,
                                           "relation": SAME_ACQUISITION}
    morph_entry = results["enhanced"].record["input"]["channels"][0]
    assert morph_entry["prepare_channel"] == "Flamingo" and morph_entry["prepare"]["function"] == "enhance_with_flamingo"
    assert morph_entry["registration"] == {"reference": "round1:ch04",
                                           "reference_sha256": hashlib.sha256(stain.tobytes()).hexdigest()}

    # Reloaded on a new FOV object after FOV.run: the record links the saved file.
    reloaded = ds.fov("FOV").run(pipeline(rotation=90)).load_registered_round("reference_stain", checkpoints=checkpoints)
    reloaded.segment(plan(STAIN_INPUT, level))
    (entry,) = reloaded.segmentation_results["nucleus"].record["input"]["channels"]
    path = "other_rounds/reference_stain/image.ome.tif"
    assert entry["sha256"] == stain_entry["sha256"]
    assert entry["registration"] == dict(stain_entry["registration"], saved={
        "path": path, "sha256": hashlib.sha256((tmp_path / "ck" / "FOV" / path).read_bytes()).hexdigest()})

    # A round that is loaded but not prepared still raises.
    unprepared = ds.fov("FOV").run(pipeline(rotation=90))
    unprepared.load_images(rounds=["reference_stain", "morph"], rotation_degrees=90)
    for channel in (STAIN_INPUT, ENHANCED_INPUT):
        with pytest.raises(ValueError, match="has no registration entry"):
            unprepared.segment(plan(channel, level))
    assert unprepared.segmentation_results == {}
