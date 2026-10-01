"""Spotiflow and Piscis without their extras: configs, dependency errors, the run order, the extracted weights
files and the records plumbing (W-272; docs/spot-finding-contract.md, "Registry entries", "Configs of the new
methods", "Pretrained weights"; checks S9 and S14).

Default tier: the extras are simulated missing by patching imports, or present as empty stand-ins, so no test
here needs the libraries, torch or the weights. Bounds: the dependency, weights and model-name errors, their
order and scale fixed at 1 are provisional contract rules without a W-266 reference (S14, S9).
"""
from dataclasses import dataclass, field, replace
import hashlib
import json
import sys
import types

import jsonschema
import numpy as np
import pandas as pd
import pytest

from starfinder.__main__ import main
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.image import ImageMetadata
from starfinder.spot_finding import (KNOWN_WEIGHTS, SPOT_FINDING_METHODS, ChannelOverride, MissingWeightsError,
    PiscisConfig, SpotFindingBackendUnavailableError, SpotFindingPlan, SpotFindingSpec, SpotiflowConfig, WeightsFile,
    WeightsHashMismatchError, find_spots, resolve_weights)
from starfinder.spot_finding import _methods
from starfinder.spot_finding._weights import weights_artifacts

from .learned_detectors import one_thread_environment, run_python
from .test_spot_finding_golden import CHANNELS, fov_with_fixture, golden_dataset
from .test_spot_finding_workflow_key import SCHEMA, detection

META = ImageMetadata("learned")
NAMESPACE = "learned/test"
EXTRAS = {SpotiflowConfig: ("spotiflow", SpotiflowConfig("smfish_3d"), np.ones((8, 16, 16), np.uint16)),
          PiscisConfig: ("piscis", PiscisConfig("20251212"), np.ones((2, 16, 16), np.uint16))}
# Every file of the four Spotiflow archives (path, SHA-256, bytes), hashed from the archives whose SHA-256 and
# MD5 match the table; the operator's fetched copies and the W-266 spike copies give the same values.
EXTRACTED = {
    "synth_3d": {"best.pt": "846d1ef438f872d50f160ad6cfb8be5b99f3f837600c52843a1391ca4eae7d4c",
                 "config.yaml": "09bd576178014753853c4dab7066fd300b3ed539e053ddaa1b4192f89a00131d",
                 "last.pt": "11a0ef1b3dc11c9d0a9a497bed301cef24f2e43c740b02f8db74fadd75cd9227",
                 "thresholds.yaml": "f63b656f20f5c45143686c34d04f20a08c078f6e5b88ad2e0f40aab0de24f04f",
                 "train_config.yaml": "37909c8acf90b827b33a27daf9d0c08ea2af2ce297e0b7ad4d494a04a20234c0"},
    "smfish_3d": {"best.pt": "1fdfd62c89a007094870782c27da163052ed72952a72d29f12e6730514d3ad6d",
                  "config.yaml": "09bd576178014753853c4dab7066fd300b3ed539e053ddaa1b4192f89a00131d",
                  "last.pt": "bf1d5f13928c09a4b03c80fed41fd292a150183c28f4926a810688bd82dc4a18",
                  "thresholds.yaml": "daa3d82d9bc79141e904bcab6ea2f24057b966b7ad43b8053383d4d7f7c4e6e5",
                  "train_config.yaml": "37909c8acf90b827b33a27daf9d0c08ea2af2ce297e0b7ad4d494a04a20234c0"},
    "general": {"best.pt": "1c3575464d621924b27f4deb66495b807f175a0ccd995d3533403f00daf806f2",
                "config.yaml": "507b60fab22c08e8819da3f7c84d8a85b3fdce4caca14c18d101b502109b0649",
                "last.pt": "2d1c04545f2205de46d7002c460d492cebe0a5c76840b494668b3ede904651fc",
                "thresholds.yaml": "4c054d3cbb2825df7a4672a63ef454fbc25a7a01e53faefc431793068ac6367e",
                "train_config.yaml": "fa805d53d763da34b68ddbaf6fd6876ff0d83d7565c4280f8b87ee14b7cd9777"},
    "hybiss": {"best.pt": "fa5d5cb313bcfd75527f3d0962ccc3654bc7aa33359097d16f3b0835b0103bd1",
               "config.yaml": "10253c542bcc21348a1a5b3810de1950be7250be51147f96a8c679eef3cc583a",
               "last.pt": "2b309de99caf9a948b922fa88e99b5599b335b77b246fd0f5d4ea2ec55f89632",
               "thresholds.yaml": "62dee75197686a9ea85fe0bf493f8ec2bcefbea20ef2748948d437b627e44045",
               "train_config.yaml": "480f56279201137dc1ec74681dfd160b2d98af12d7569b8fdaad667e45bbc7cd"},
}


def detect(image, config):
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(root))
    return root


def stand_ins(monkeypatch, *names):
    """Empty modules in place of the named imports: require() finds them, and any attribute use fails."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


# --- Without the extras ---------------------------------------------------------------------------------

def test_without_the_extras_importing_and_constructing_work_and_running_raises(empty_cache):
    code = f"""
import importlib.abc, sys
class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "spotiflow", "piscis"):
            raise ImportError(f"No module named {{name!r}}")
sys.meta_path.insert(0, Missing())
import numpy as np
from starfinder.image import ImageMetadata
from starfinder.spot_finding import PiscisConfig, SpotFindingBackendUnavailableError, SpotiflowConfig, find_spots
for config, shape in ((SpotiflowConfig("smfish_3d"), (8, 16, 16)), (PiscisConfig("20251212"), (2, 16, 16))):
    try:
        find_spots(np.ones(shape, np.uint16), config=config, metadata=ImageMetadata("x"), spot_namespace="x")
    except SpotFindingBackendUnavailableError as error:
        print(error)
print(sorted(m for m in ("torch", "spotiflow", "piscis") if m in sys.modules))
"""
    out = run_python(code, one_thread_environment(STARFINDER_WEIGHTS_DIR=str(empty_cache))).splitlines()
    assert out == ["spot-finding method 'spotiflow' requires spotiflow; install the 'spotiflow' extra "
                   "(starfinder[spotiflow])",
                   "spot-finding method 'piscis' requires piscis; install the 'piscis' extra (starfinder[piscis])",
                   "[]"]
    assert not empty_cache.exists()


@pytest.mark.parametrize("config_type", [SpotiflowConfig, PiscisConfig])
@pytest.mark.parametrize("missing", ["library", "torch"])
def test_the_dependency_error_names_the_module_and_the_extra(config_type, missing, monkeypatch, empty_cache):
    extra, config, image = EXTRAS[config_type]
    if missing == "library":
        monkeypatch.setitem(sys.modules, extra, None)
    else:
        stand_ins(monkeypatch, extra)
        monkeypatch.setitem(sys.modules, "torch", None)
    module = extra if missing == "library" else "torch"
    expected = (f"spot-finding method '{extra}' requires {module}; install the '{extra}' extra "
                f"(starfinder[{extra}])")
    # The weights are missing too: require() runs first, so the dependency error is the one raised.
    with pytest.raises(SpotFindingBackendUnavailableError) as raised:
        detect(image, config)
    assert isinstance(raised.value, ImportError) and str(raised.value) == expected
    monkeypatch.setattr(_methods, "_PY314", True)
    with pytest.raises(SpotFindingBackendUnavailableError) as raised:
        detect(image, config)
    assert str(raised.value) == f"{expected} (the extra is not available on Python 3.14 and later)"


@pytest.mark.parametrize("config_type", [SpotiflowConfig, PiscisConfig])
def test_with_the_extras_found_missing_weights_raise_before_any_model_is_built(config_type, monkeypatch,
                                                                                empty_cache):
    extra, config, image = EXTRAS[config_type]
    stand_ins(monkeypatch, extra, "torch")   # the stand-ins have no model class: building one would fail
    with pytest.raises(MissingWeightsError) as raised:
        detect(image, config)
    entry = KNOWN_WEIGHTS[(extra, config.model)]
    assert str(empty_cache / extra / config.model / entry.files[0].path) in str(raised.value)
    assert f"'starfinder weights fetch {extra} {config.model}'" in str(raised.value)


# --- Configs ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("config_type, model, known", [
    (SpotiflowConfig, "latest", "['general', 'hybiss', 'smfish_3d', 'synth_3d']"),
    (SpotiflowConfig, "20251212", "['general', 'hybiss', 'smfish_3d', 'synth_3d']"),
    (PiscisConfig, "smfish_3d", "['20230905', '20251212']"),
    (PiscisConfig, "20230905.pt", "['20230905', '20251212']")])
def test_an_unknown_model_or_one_of_the_other_method_lists_the_known_names(config_type, model, known):
    with pytest.raises(ValueError) as raised:
        config_type(model)
    assert str(raised.value).endswith(f"known models: {known}")


def test_every_run_names_its_weights():
    with pytest.raises(TypeError):
        SpotiflowConfig()
    with pytest.raises(TypeError):
        PiscisConfig()
    for config_type in (SpotiflowConfig, PiscisConfig):
        with pytest.raises(ValueError):
            config_type(None)


@pytest.mark.parametrize("config", [SpotiflowConfig("smfish_3d"), PiscisConfig("20230905")])
@pytest.mark.parametrize("scale", [2, 0.5, 2.0, True, float("nan")])
def test_s9_a_scale_other_than_one_raises_at_construction(config, scale):
    with pytest.raises(ValueError, match="scale must be 1"):
        replace(config, scale=scale)
    assert replace(config, scale=1).scale == 1


def test_the_native_defaults():
    assert (SpotiflowConfig("general").prob_thresh, SpotiflowConfig("general").min_distance,
            SpotiflowConfig("general").exclude_border, SpotiflowConfig("general").subpix,
            SpotiflowConfig("general").n_tiles, SpotiflowConfig("general").scale) == (None, 1, False, None, None, 1.0)
    assert (PiscisConfig("20251212").threshold, PiscisConfig("20251212").min_distance,
            PiscisConfig("20251212").input_size, PiscisConfig("20251212").scale) == (0.5, 1, None, 1.0)
    assert (SpotiflowConfig("hybiss").method, PiscisConfig("20251212").method) == ("spotiflow", "piscis")


@pytest.mark.parametrize("config, change", [
    (SpotiflowConfig("smfish_3d"), dict(prob_thresh=1.5)), (SpotiflowConfig("smfish_3d"), dict(prob_thresh=-0.1)),
    (SpotiflowConfig("smfish_3d"), dict(min_distance=0)), (SpotiflowConfig("smfish_3d"), dict(min_distance=1.5)),
    (SpotiflowConfig("smfish_3d"), dict(exclude_border=1)), (SpotiflowConfig("smfish_3d"), dict(subpix=0)),
    (SpotiflowConfig("smfish_3d"), dict(n_tiles=(2, 2))), (SpotiflowConfig("general"), dict(n_tiles=(1, 2, 2))),
    (SpotiflowConfig("general"), dict(n_tiles=(0, 2))), (SpotiflowConfig("general"), dict(n_tiles=[2, 2])),
    (SpotiflowConfig("general"), dict(channel_labels=("a", "a"))),
    (PiscisConfig("20251212"), dict(threshold=1.5)), (PiscisConfig("20251212"), dict(threshold=float("nan"))),
    (PiscisConfig("20251212"), dict(min_distance=0)), (PiscisConfig("20251212"), dict(input_size=0)),
    (PiscisConfig("20251212"), dict(input_size=32.0)), (PiscisConfig("20251212"), dict(input_size=True))])
def test_invalid_settings_raise_at_construction(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


# --- The extracted files of the Spotiflow archives -------------------------------------------------------

def test_the_table_holds_every_extracted_file_of_the_spotiflow_archives():
    for (method, model), entry in KNOWN_WEIGHTS.items():
        if method == "piscis":
            assert entry.extracted == ()
            continue
        assert {item.path: item.sha256 for item in entry.extracted} == EXTRACTED[model]
        assert entry.files[0] in entry.extracted
        sizes = {item.path: item.bytes for item in entry.extracted}
        assert sizes["last.pt"] == sizes["best.pt"] == entry.files[0].bytes


@pytest.fixture
def folder_fixture(empty_cache, monkeypatch):
    """A Spotiflow-shaped fixture model in the cache: best.pt, config.yaml, thresholds.yaml and last.pt."""
    contents = {"best.pt": b"best\n", "config.yaml": b"is_3d: true\n", "thresholds.yaml": b"prob_thresh_best: 0.4\n",
                "last.pt": b"last\n"}
    folder = empty_cache / "spotiflow" / "fixture"
    folder.mkdir(parents=True)
    files = {}
    for name, data in contents.items():
        (folder / name).write_bytes(data)
        files[name] = WeightsFile(name, hashlib.sha256(data).hexdigest(), len(data))
    entry = replace(KNOWN_WEIGHTS[("spotiflow", "smfish_3d")], model="fixture", revision="fixture",
                    files=(files["best.pt"],), extracted=tuple(files[n] for n in sorted(files)))
    monkeypatch.setitem(KNOWN_WEIGHTS, ("spotiflow", "fixture"), entry)
    return folder


def test_resolve_weights_checks_the_named_extracted_files(folder_fixture):
    reads = ("config.yaml", "thresholds.yaml")
    assert resolve_weights("spotiflow", "fixture", extracted=reads) == folder_fixture
    assert [a["path"] for a in weights_artifacts("spotiflow", "fixture", folder_fixture, reads)] == [
        str((folder_fixture / name).resolve()) for name in ("best.pt", "config.yaml", "thresholds.yaml")]
    (folder_fixture / "config.yaml").write_bytes(b"is_3d: false\n")
    assert resolve_weights("spotiflow", "fixture") == folder_fixture   # not named: not checked
    with pytest.raises(WeightsHashMismatchError, match=hashlib.sha256(b"is_3d: false\n").hexdigest()):
        resolve_weights("spotiflow", "fixture", extracted=reads)
    (folder_fixture / "thresholds.yaml").unlink()
    with pytest.raises(MissingWeightsError, match="thresholds.yaml"):
        resolve_weights("spotiflow", "fixture", extracted=("thresholds.yaml",))
    with pytest.raises(ValueError, match="extracts no"):
        resolve_weights("spotiflow", "fixture", extracted=("model.yaml",))


def test_the_verify_command_rehashes_every_extracted_file(folder_fixture, capsys):
    assert main(["weights", "verify", "spotiflow", "fixture"]) == 0
    assert "spotiflow\tfixture\tverified" in capsys.readouterr().out
    (folder_fixture / "last.pt").write_bytes(b"changed\n")
    assert main(["weights", "verify", "spotiflow", "fixture"]) == 1
    assert "last.pt" in capsys.readouterr().out


# --- Records: resolved settings, the model record and run.json -------------------------------------------

@dataclass(frozen=True)
class RecordingConfig:
    """A fixture learned method: one spot per channel; None settings resolve to 0.25 and a named model."""
    model: str = "a"
    level: float | None = None
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="recording_detector", init=False)

    def __post_init__(self):
        pass


def recording_run(image, config, context):
    rows = [(1.0, 2.0, 3.0, c) for c in context.channels]
    table = pd.DataFrame(rows, columns=["z", "y", "x", "channel"]).astype(
        {"z": "float64", "y": "float64", "x": "float64", "channel": "int64"})
    effective = replace(config, level=0.25 if config.level is None else config.level)
    artifact = {"name": f"recording/{config.model}", "path": f"/weights/{config.model}.pt", "sha256": config.model * 64,
                "source": "file:///source", "revision": "r1"}
    return table, {"thresholds": tuple(effective.level for _ in context.channels),
                   "effective": tuple(effective for _ in context.channels),
                   "model": {"method": "recording", "model": config.model, "artifacts": [artifact],
                             "training_pixel_size": "1 um", "training_pixel_size_provenance": "fixture"}}


@pytest.fixture
def recording(monkeypatch):
    spec = SpotFindingSpec("recording_detector", recording_run, pipeline=True, dimensions=frozenset({2, 3}),
                           output_columns=("z", "y", "x", "channel"), weights=True)
    monkeypatch.setitem(SPOT_FINDING_METHODS, RecordingConfig, spec)


def test_effective_settings_and_the_model_record_follow_the_method(recording):
    config = RecordingConfig(channel_labels=CHANNELS)
    plan = SpotFindingPlan(config, (ChannelOverride("ch02", replace(config, model="b", level=0.5)),
                                    ChannelOverride("ch03", replace(config, level=0.75))))
    diagnostics = detect(np.ones((2, 8, 8, 4), np.uint16), plan).diagnostics
    assert {k: v["level"] for k, v in diagnostics["effective_settings"].items()} == {
        "ch00": 0.25, "ch01": 0.25, "ch02": 0.5, "ch03": 0.75}
    assert diagnostics["thresholds"] == (0.25, 0.25, 0.5, 0.75)
    model = diagnostics["model"]
    assert (model["model"], [a["path"] for a in model["artifacts"]]) == ("a", ["/weights/a.pt"])
    assert list(model["channel_overrides"]) == ["ch02"] and model["channel_overrides"]["ch02"]["model"] == "b"


def test_run_json_records_the_loaded_files_as_artifacts(recording, tmp_path):
    checkpoints = CheckpointConfig(stages=("candidates",), directory=tmp_path / "checkpoints")
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d").run(PipelineConfig(detection=RecordingConfig()),
                                                               checkpoints=checkpoints)
    data = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    (entry,) = [step for step in data["steps"] if step["name"] == "find_spots"][0]["methods"]
    assert entry["artifacts"] == fov.spot_result.diagnostics["model"]["artifacts"]
    assert entry["artifacts"][0]["path"] == "/weights/a.pt"


# --- Workflow keys -----------------------------------------------------------------------------------------

def test_the_workflow_block_selects_the_learned_methods():
    block = {"method": "spotiflow", "model": "smfish_3d", "prob_thresh": None, "min_distance": 2, "n_tiles": [1, 2, 2],
             "channel_overrides": {"ch03": {"prob_thresh": 0.5}}}
    config = SpotiflowConfig("smfish_3d", min_distance=2, n_tiles=(1, 2, 2))
    assert detection(block) == SpotFindingPlan(config, (ChannelOverride("ch03", replace(config, prob_thresh=0.5)),))
    assert detection({"method": "piscis", "model": "20230905", "threshold": 0.6, "input_size": 128}) == PiscisConfig(
        "20230905", threshold=0.6, input_size=128)
    with pytest.raises(ValueError, match="requires model"):
        detection({"method": "piscis"})
    for key, value in (("intensity_threshold", 5.0), ("min_distance_voxels", 2)):
        with pytest.raises(ValueError, match=key):
            detection({"method": "spotiflow", "model": "general", key: value})
    schema = {"$defs": SCHEMA["$defs"], "$ref": "#/$defs/spot_finding_params"}
    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(dict(block, run=True)))
    assert list(validator.iter_errors({"run": True, "method": "piscis", "model": "20251212", "scale": 2}))
