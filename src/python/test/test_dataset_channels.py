"""Channel name and wavelength per round: ChannelInfo, Dataset.channel_index, its users, the workflow translation
and the records (W-336; docs/coordination.md, "Channels")."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml
from skimage.measure import label

from starfinder.dataset import (ChannelInfo, CheckpointConfig, Dataset, PipelineConfig, RegistrationRecipe,
                                RegistrationStep, RoundState, from_workflow_config)
from starfinder.dataset.workflow import _nuclei_registration
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.registration import RegistrationSignalConfig, TranslationConfig
from starfinder.segmentation import (SEGMENTATION_METHODS, FlamingoEnhancementConfig, InputChannel, SegmentationPlan,
                                     SegmentationRun, SegmentationSpec)
from starfinder.synthetic import development_scene_preset, generate_formed_scene

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
SEQUENCING = ("ch00", "ch01", "ch02", "ch03")
MORPH = (ChannelInfo("ch00", "Flamingo", 488), ChannelInfo("ch01", "RBD", 561), ChannelInfo("ch02", "DAPI"))
STAIN = ChannelInfo("ch04", "DAPI", 405)


def dataset(channel_order=tuple(ChannelInfo(c, "seq") for c in SEQUENCING), other=None, stains=(STAIN,), **kwargs):
    """round1 (reference) and round2 sequencing, morph an other round with Flamingo, RBD and DAPI, one stain."""
    rounds = RoundState(["round1", "round2"], ["morph"], "round1")
    return Dataset("in", "out", "data", "sample", "out", rounds, channel_order,
                   other_channel_order={"morph": MORPH} if other is None else other, reference_stains=stains, **kwargs)


# --- Type and forms -------------------------------------------------------------------------------------

def test_channel_info_from_a_string_three_values_and_a_mapping_are_equal():
    assert ChannelInfo.from_value("ch00") == ChannelInfo("ch00", None, None) == ChannelInfo.from_value(
        {"channel": "ch00"})
    named = ChannelInfo("ch02", "DAPI", 405)
    assert ChannelInfo.from_value({"wavelength": 405, "channel": "ch02", "name": "DAPI"}) == named
    assert ChannelInfo.from_value(named) is named
    assert named.wavelength == 405.0 and isinstance(named.wavelength, float)
    # The written form of a missing wavelength reads back as None.
    assert ChannelInfo.from_value({"channel": "ch02", "name": "DAPI", "wavelength": "unavailable"}) == \
        ChannelInfo("ch02", "DAPI")


def test_a_missing_wavelength_is_written_as_unavailable():
    assert ChannelInfo("ch00", "seq").record() == {"channel": "ch00", "name": "seq", "wavelength": "unavailable"}
    assert ChannelInfo("ch00", None, 488).record() == {"channel": "ch00", "name": None, "wavelength": 488.0}
    record = dataset().channel_record()
    assert list(record) == ["round1", "round2", "morph", "reference_stain"]
    assert record["morph"] == [{"channel": "ch00", "name": "Flamingo", "wavelength": 488.0},
                               {"channel": "ch01", "name": "RBD", "wavelength": 561.0},
                               {"channel": "ch02", "name": "DAPI", "wavelength": "unavailable"}]
    assert record["reference_stain"] == [{"channel": "ch04", "name": "DAPI", "wavelength": 405.0}]
    assert "reference_stain" not in dataset(stains=()).channel_record()


@pytest.mark.parametrize("form", ["string", "info", "mapping"])
def test_each_accepted_form_gives_the_same_channel_information(form):
    def convert(info):
        return {"string": info.channel, "info": info,
                "mapping": {k: v for k, v in (("channel", info.channel), ("name", info.name),
                                              ("wavelength", info.wavelength)) if v is not None}}[form]

    plain = tuple(ChannelInfo(c) for c in SEQUENCING)
    morph = MORPH if form != "string" else tuple(ChannelInfo(c.channel) for c in MORPH)
    stains = (STAIN,) if form != "string" else (ChannelInfo("ch04"),)
    ds = dataset([convert(c) for c in plain], {"morph": [convert(c) for c in morph]}, [convert(c) for c in stains])
    assert ds.channel_info("round1") == ds.channel_info("round2") == plain
    assert ds.channel_info("morph") == morph and ds.channel_info("reference_stain") == stains
    # channel_labels and the fields keep today's patterns.
    assert ds.channel_order == SEQUENCING and ds.other_channel_order == {"morph": ("ch00", "ch01", "ch02")}
    assert ds.channel_labels("round1") == SEQUENCING and ds.channel_labels("morph") == ("ch00", "ch01", "ch02")
    assert ds.channel_labels("reference_stain") == ("ch04",)


def test_the_string_form_keeps_todays_labels():
    ds = Dataset("in", "out", "data", "sample", "out", RoundState(["round1"], ["morph"], "round1"), list(SEQUENCING),
                 other_channel_order={"morph": ["a", "b"]})
    assert ds.channel_order == SEQUENCING and ds.other_channel_order == {"morph": ("a", "b")}
    assert ds.channel_labels("round1") == SEQUENCING and ds.channel_labels("morph") == ("a", "b")
    assert ds.reference_stains == () and ds.channel_info("reference_stain") == ()
    assert ds.channel_info("morph") == (ChannelInfo("a"), ChannelInfo("b"))


@pytest.mark.parametrize("change, error", [
    (dict(channel_order=[1, "ch01"]), TypeError),
    (dict(channel_order=[None]), TypeError),
    (dict(channel_order=[""]), ValueError),
    (dict(channel_order=[{"name": "seq"}]), ValueError),
    (dict(channel_order=[{"channel": "ch00", "colour": "red"}]), ValueError),
    (dict(channel_order="ch00"), TypeError),
    (dict(channel_order=["ch00", ChannelInfo("ch00", "seq")]), ValueError),
    (dict(other={"morph": ["ch00", "ch00"]}), ValueError),
    (dict(other={"morph": []}), ValueError),
    (dict(stains=["ch04", {"channel": "ch04"}]), ValueError),
    (dict(stains=["ch00"]), ValueError),           # a reference stain is not a sequencing colour
])
def test_invalid_channels_raise(change, error):
    kwargs = dict(change)
    with pytest.raises(error):
        dataset(**kwargs)


@pytest.mark.parametrize("wavelength", [0, -488, math.nan, math.inf, -math.inf])
def test_a_non_positive_or_non_finite_wavelength_raises(wavelength):
    with pytest.raises(ValueError, match="positive and finite"):
        ChannelInfo("ch00", "seq", wavelength)
    with pytest.raises(ValueError, match="positive and finite"):
        dataset(channel_order=[{"channel": "ch00", "wavelength": wavelength}])


@pytest.mark.parametrize("value", [True, "488", [488]])
def test_a_wavelength_that_is_not_a_number_raises(value):
    with pytest.raises(TypeError):
        ChannelInfo("ch00", wavelength=value)


def test_an_empty_or_non_string_name_raises():
    with pytest.raises(ValueError):
        ChannelInfo("ch00", "")
    with pytest.raises(TypeError):
        ChannelInfo("ch00", 3)


@pytest.mark.parametrize("rounds", [RoundState(["round1", "reference_stain"], reference_round="round1"),
                                    RoundState(["round1"], ["reference_stain"], "round1")])
def test_a_round_named_reference_stain_raises(rounds):
    with pytest.raises(ValueError, match="reserved"):
        Dataset("in", "out", "data", "sample", "out", rounds, SEQUENCING)


def test_reference_stains_need_a_reference_round():
    with pytest.raises(ValueError, match="no reference round"):
        Dataset("in", "out", "data", "sample", "out", RoundState(["round1"]), SEQUENCING, reference_stains=[STAIN])


# --- The lookup rule --------------------------------------------------------------------------------------

def test_channel_index_finds_a_pattern_a_unique_name_and_an_index():
    ds = dataset()
    assert ds.channel_index("morph", "ch02") == ds.channel_index("morph", "DAPI") == 2
    assert ds.channel_index("morph", "Flamingo") == 0 and ds.channel_index("morph", "RBD") == 1
    assert ds.channel_index("round2", "ch03") == 3 and ds.channel_index("round2", 1) == 1
    assert ds.channel_index("morph", np.int64(2)) == 2
    assert ds.channel_index("reference_stain", "DAPI") == ds.channel_index("reference_stain", "ch04") == 0


@pytest.mark.parametrize("round_name, key, message", [
    ("round1", "seq", "round 'round1' has 4 channels named 'seq'"),
    ("morph", "Nissl", "round 'morph' has no channel 'Nissl'"),
    ("morph", 3, "channel index 3 is outside round 'morph'"),
    ("morph", -1, "channel index -1 is outside round 'morph'"),
    ("reference_stain", "PI", "round 'reference_stain' has no channel 'PI'"),
])
def test_channel_index_names_the_round_and_its_channels(round_name, key, message):
    ds = dataset()
    with pytest.raises(ValueError, match=message) as error:
        ds.channel_index(round_name, key)
    listed = ", ".join(f"{c.channel} ({c.name})" for c in ds.channel_info(round_name))
    assert str(error.value).endswith(f"its channels are {listed}")


def test_a_pattern_wins_over_the_name_of_another_channel():
    ds = dataset(other={"morph": [ChannelInfo("DAPI", "Flamingo"), ChannelInfo("ch01", "DAPI")]})
    assert ds.channel_index("morph", "DAPI") == 0 and ds.channel_index("morph", "Flamingo") == 0


def test_channel_index_refuses_an_unknown_round_and_a_key_of_another_type():
    ds = dataset()
    with pytest.raises(ValueError, match="not a configured round"):
        ds.channel_index("round9", "ch00")
    for key in (True, 1.0, None):
        with pytest.raises(TypeError):
            ds.channel_index("morph", key)


# --- Users of the rule: FOV.register_rounds and FOV.segment ----------------------------------------------

@dataclass(frozen=True)
class ThresholdConfig:
    """A test-only method: the connected components of channel 0 above level."""
    level: float = 0.0
    method: str = field(default="w336_threshold", init=False)


def _threshold(image, config, context):
    labels = label(image[..., 0] > config.level, connectivity=1).astype(np.int32)
    return labels, {"effective": {"level": config.level}}


THRESHOLD = SegmentationSpec("w336_threshold", _threshold, targets=frozenset({"nucleus"}),
                             roles=frozenset({"nuclear", "amplicon"}), required_roles=(frozenset({"nuclear"}),),
                             seeds="optional", dimensions=frozenset({2, 3}), models=False, devices=frozenset({"cpu"}))
SPACING = dict(spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")


@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setitem(SEGMENTATION_METHODS, ThresholdConfig, THRESHOLD)


@pytest.fixture(scope="module")
def development(tmp_path_factory):
    """The development preset (small, clean) written as TIFFs, with spacing."""
    book, config = development_scene_preset("clean", size="small")
    scene = generate_formed_scene(book, config=config)
    root = tmp_path_factory.mktemp("development")
    for name, image in scene.rounds.items():
        metadata = replace(scene.round_metadata[name], **SPACING)
        for c, channel in enumerate(scene.channel_labels):
            save_volume(image[..., c], root / name / "FOV_001" / f"{channel}.tif", metadata=metadata)
    return root, scene


def morphology_fov(development):
    """The reference round loaded and morph (its first three channels, displaced by (0, 2, −1)) placed beside it.

    morph is a configured other round with Flamingo, RBD and DAPI; its DAPI is the reference round's channel 2.
    """
    root, scene = development
    ref = scene.round_labels[0]
    channels = [ChannelInfo(c, "seq", w) for c, w in zip(scene.channel_labels, (488, None, 594, 647))]
    rounds = RoundState(list(scene.round_labels), ["morph"], ref)
    ds = Dataset(root, root / "out", "dev", "sample", "out", rounds, channels, other_channel_order={"morph": MORPH})
    fov = ds.fov("FOV_001").load_images(rounds=[ref])
    fov.images["morph"] = np.ascontiguousarray(np.roll(fov.images[ref][..., :3], (0, 2, -1), axis=(0, 1, 2)))
    fov.metadata["morph"] = replace(fov.metadata[ref], frame_id="morph")
    return fov


def registered(development, moving):
    fov = morphology_fov(development)
    signal = RegistrationSignalConfig("channel", reference_channel=fov.dataset.channel_labels(
        fov.rounds.reference_round)[2], moving_channel=moving)
    return fov.register_rounds(RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=signal),
                               rounds=["morph"])


def test_registration_resolves_the_same_channel_by_name_and_by_pattern(development):
    by_name, by_pattern, by_index = registered(development, "DAPI"), registered(development, "ch02"), \
        registered(development, 2)
    shifts = [fov.registration_results["morph"][0].transform for fov in (by_name, by_pattern, by_index)]
    assert shifts[0] == shifts[1] == shifts[2]
    assert np.array_equal(by_name.images["morph"], by_pattern.images["morph"])


def test_segmentation_resolves_the_same_channel_by_name_and_by_pattern(development, threshold):
    fov = registered(development, "DAPI")
    level = float(np.mean(fov.images["morph"][..., 2]))
    results = {}
    for key in ("DAPI", "ch02", 2):
        fov.segment(SegmentationPlan((SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", "morph", key),),
                                                      ThresholdConfig(level)),)))
        results[key] = fov.segmentation_results["nucleus"]
    digests = {key: result.record["input"]["sha256"] for key, result in results.items()}
    assert digests["DAPI"] == digests["ch02"] == digests[2]
    assert np.array_equal(results["DAPI"].labels, results["ch02"].labels)
    (entry,) = results["DAPI"].record["input"]["channels"]
    assert (entry["channel"], entry["name"], entry["wavelength"]) == ("DAPI", "DAPI", "unavailable")
    (entry,) = results[2].record["input"]["channels"]
    assert (entry["channel"], entry["name"], entry["wavelength"]) == (2, "DAPI", "unavailable")


def test_an_unknown_name_gives_one_error_form(development, threshold):
    fov = registered(development, "DAPI")
    messages = []
    with pytest.raises(ValueError) as error:
        registered(development, "Nissl")
    messages.append(str(error.value))
    with pytest.raises(ValueError) as error:
        fov.segment(SegmentationPlan((SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", "morph", "Nissl"),),
                                                      ThresholdConfig()),)))
    messages.append(str(error.value))
    with pytest.raises(ValueError) as error:
        fov.segment(SegmentationPlan((SegmentationRun("nucleus", "nucleus", (InputChannel(
            "nuclear", "morph", "DAPI", prepare=FlamingoEnhancementConfig(), prepare_channel="Nissl"),),
            ThresholdConfig()),)))
    messages.append(str(error.value))
    with pytest.raises(ValueError) as expected:
        fov.dataset.channel_index("morph", "Nissl")
    assert messages == [str(expected.value)] * 3
    assert messages[0].startswith("round 'morph' has no channel 'Nissl'")
    assert fov.segmentation_results == {}


def test_no_second_lookup_is_left_in_the_plan_or_the_fov():
    package = ROOT / "src/python/starfinder"
    plan, fov = (package / "segmentation/_plan.py").read_text(), (package / "dataset/fov.py").read_text()
    for text in (plan, fov):
        assert "other_channel_order.get(" not in text
        assert "_resident_channel_index(" in text


# --- Workflow translation ---------------------------------------------------------------------------------

def full_config(**change):
    config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
    config.update(change)
    return config


OBJECTS = [{"wavelength": 488, "channel": "ch00", "name": "seq"}, {"wavelength": 546, "channel": "ch02", "name": "seq"},
           {"wavelength": 594, "channel": "ch01", "name": "seq"}, {"wavelength": 647, "channel": "ch03", "name": "seq"}]
ADDITIONAL = [{"round_name": "morph", "channel_order": [
    {"wavelength": 488, "channel": "ch00", "name": "Flamingo"}, {"channel": "ch01", "name": "RBD"},
    {"wavelength": 405, "channel": "ch02", "name": "DAPI"}]}]


@pytest.mark.workflow
def test_both_forms_of_seq_channel_order_give_the_same_patterns():
    patterns = from_workflow_config(full_config()).dataset
    objects = from_workflow_config(full_config(seq_channel_order=OBJECTS)).dataset
    bare = from_workflow_config(full_config(seq_channel_order=[{"channel": c} for c in patterns.channel_order])).dataset
    assert patterns.channel_order == objects.channel_order == ("ch00", "ch02", "ch01", "ch03")
    assert bare.channel_info("round1") == patterns.channel_info("round1")
    assert objects.channel_info("round1") == tuple(ChannelInfo.from_value(o) for o in OBJECTS)
    adapted = (from_workflow_config(full_config()), from_workflow_config(full_config(seq_channel_order=OBJECTS)))
    assert adapted[0].pipeline.load == adapted[1].pipeline.load
    for config in (full_config(), full_config(seq_channel_order=OBJECTS), full_config(additional_round=ADDITIONAL),
                   full_config(seq_channel_order=OBJECTS, additional_round=ADDITIONAL)):
        jsonschema.validate(config, SCHEMA)


@pytest.mark.workflow
def test_every_rule_names_the_other_rounds_with_their_channels():
    config = full_config(seq_channel_order=OBJECTS, additional_round=ADDITIONAL)
    sequencing = from_workflow_config(config).dataset
    nuclei, stains, folders = _nuclei_registration(config)
    expected = tuple(ChannelInfo.from_value(c) for c in ADDITIONAL[0]["channel_order"])
    for ds in (sequencing, nuclei):
        assert ds.rounds.other_rounds == ["morph"]
        assert ds.channel_info("morph") == expected and ds.channel_labels("morph") == ("ch00", "ch01", "ch02")
        assert ds.channel_info("round1") == tuple(ChannelInfo.from_value(o) for o in OBJECTS)
        assert ds.reference_stains == (ChannelInfo("ch04", "DAPI"),)
    # The legacy ref_channel match stays MATLAB's substring match.
    assert stains == {"morph": "ch02"} and folders == {"morph": ("Flamingo", "RBD", "DAPI")}
    nuclei, stains, _ = _nuclei_registration(dict(config, ref_channel="DAP"))
    assert stains == {"morph": "ch02"}
    # An entry without channel_order takes the sequencing channels in the sequencing rule.
    plain = from_workflow_config(full_config(additional_round=[{"round_name": "morph"}])).dataset
    assert plain.channel_labels("morph") == plain.channel_order


@pytest.mark.workflow
def test_dapi_round_gives_the_reference_stain():
    assert from_workflow_config(full_config()).dataset.reference_stains == (ChannelInfo("ch04", "DAPI"),)
    without = full_config()
    del without["dapi_round"]
    assert from_workflow_config(without).dataset.reference_stains == ()
    config = dict(without, additional_round=ADDITIONAL)
    assert _nuclei_registration(config)[0].reference_stains == ()


@pytest.mark.workflow
def test_a_dapi_round_other_than_the_reference_round_raises():
    config = full_config(dapi_round="round2", additional_round=ADDITIONAL)
    for translate in (from_workflow_config, _nuclei_registration):
        with pytest.raises(ValueError, match="dapi_round 'round2' differs from ref_round 'round1'"):
            translate(config)


@pytest.mark.workflow
def test_invalid_shared_channel_entries_raise():
    with pytest.raises(ValueError, match="round_name"):
        from_workflow_config(full_config(additional_round=["morph"]))
    with pytest.raises(ValueError, match="positive and finite"):
        from_workflow_config(full_config(seq_channel_order=[dict(OBJECTS[0], wavelength=0), *OBJECTS[1:]]))
    with pytest.raises(ValueError, match="repeats the channel patterns"):
        from_workflow_config(full_config(additional_round=[{"round_name": "morph", "channel_order": ["a", "a"]}]))


# --- Records ----------------------------------------------------------------------------------------------

def test_run_json_and_segmentation_json_record_the_channels(development, threshold, tmp_path):
    root, scene = development
    ref = scene.round_labels[0]
    channels = [ChannelInfo(c, "seq", w) for c, w in zip(scene.channel_labels, (488, None, 594, 647))]
    ds = Dataset(root, root / "out", "dev", "sample", "out", RoundState(list(scene.round_labels), reference_round=ref),
                 channels, reference_stains=[STAIN])
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                              registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)))
    fov = ds.fov("FOV_001").run(pipeline, checkpoints=checkpoints)
    directory = tmp_path / "checkpoints" / "FOV_001"
    run = json.loads((directory / "run.json").read_text())
    written = [c.record() for c in channels]
    assert run["channels"] == {**{name: written for name in scene.round_labels},
                               "reference_stain": [{"channel": "ch04", "name": "DAPI", "wavelength": 405.0}]}
    assert written[1]["wavelength"] == "unavailable"

    level = float(np.mean(fov.images[ref][..., 0]))
    plan = SegmentationPlan((SegmentationRun("nucleus", "nucleus", (
        InputChannel("nuclear", ref, scene.channel_labels[1]), InputChannel("amplicon", reference_merged=True)),
        ThresholdConfig(level)),))
    fov.segment(plan, checkpoints=checkpoints)
    path = directory / "segmentation" / "nucleus" / "segmentation.json"
    record = json.loads(path.read_text())
    entries = record["input"]["channels"]
    assert [(e["role"], e["name"], e["wavelength"]) for e in entries] == [
        ("nuclear", "seq", "unavailable"), ("amplicon", None, "unavailable")]
    assert all({"channel", "name", "wavelength"} <= set(e) for e in entries)

    # Files written before W-336 (without the new keys) still load.
    del run["channels"]
    (directory / "run.json").write_text(json.dumps(run))
    for entry in entries:
        del entry["name"], entry["wavelength"]
    path.write_text(json.dumps(record))
    fresh = ds.fov("FOV_001").load_checkpoint("registered", checkpoints=checkpoints)
    assert np.array_equal(fresh.images[ref], fov.images[ref])
    loaded = fresh.load_segmentation("nucleus", checkpoints=checkpoints)
    assert np.array_equal(loaded.labels, fov.segmentation_results["nucleus"].labels)
    assert "name" not in loaded.record["input"]["channels"][0]
    assert "channels" not in json.loads((directory / "run.json").read_text())
