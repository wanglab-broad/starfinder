"""Regenerative preprocessing trials and expansion without full-size generation."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from starfinder.benchmark._experiment import expand_experiment
from starfinder.benchmark._trial import _main, _resolve
from starfinder.synthetic import UNCALIBRATED_PREPROCESSING_VERSION

pytestmark = [pytest.mark.benchmark, pytest.mark.contract]


def _input(round_label="round4", **extra):
    return dict(scene=dict(preset="uncalibrated_preprocessing", version=UNCALIBRATED_PREPROCESSING_VERSION),
                round=round_label, **extra)


def _method(recipe=False):
    minimum = dict(method="min_max_normalization", config=dict(output_dtype="uint8", output_range=[0, 255]))
    value = dict(module="preprocessing", steps=[dict(method="histogram_matching")])
    if recipe:
        value["steps"].insert(0, minimum)
        value["reference_steps"] = [minimum]
    return value


def test_scene_expansion_is_stable_and_does_not_generate(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Expansion must never generate an image")
    monkeypatch.setattr("starfinder.synthetic.generate_amplicon_scene", forbidden)
    saturation = {role: dict(scene=dict(k=2.378414230005442)) for role in ("image", "reference")}
    experiment = dict(methods=[_method(), _method(True)],
                      conditions=[dict(name="clean"), dict(name="saturation", inputs=saturation)],
                      seeds=[1, 2], inputs=[dict(image=_input(), reference=_input("round1", channel="ch00"))],
                      root_output_path=str(tmp_path / "scratch"))
    source = tmp_path / "experiment.json"
    source.write_text(json.dumps(experiment))
    first = expand_experiment(source, tmp_path / "one")
    before = {p.name: p.read_bytes() for p in (tmp_path / "one").glob("*.json")}
    assert first == expand_experiment(source, tmp_path / "one")
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "one").glob("*.json")}
    second = expand_experiment(source, tmp_path / "two")
    hashes = lambda manifest: [entry["configuration_sha256"] for entry in manifest["trials"]]
    assert hashes(first) == hashes(second)
    assert len(set(hashes(first))) == 8
    for index, item in enumerate(first["trials"]):
        config = json.loads((tmp_path / "one" / item["path"]).read_text())
        assert config["root_output_path"] == str(tmp_path / "scratch" / f"trial-{index:06d}")
        for role in ("image", "reference"):
            entry = config["inputs"][role]
            assert entry["scene"]["seed"] == config["seed"]
            assert entry["scene"]["condition"] == config["condition"]["name"]
            assert "path" not in entry
            assert entry["scene_configuration_sha256"].startswith("sha256:")
            if config["condition"]["name"] == "saturation":
                assert entry["scene"]["k"] == pytest.approx(2.378414230005442, abs=1e-12)


@pytest.mark.parametrize("recipe", [False, True])
def test_scene_trial_generates_once_selects_channel_and_normalizes_reference(tmp_path, monkeypatch, recipe):
    # Four equally populated reference levels give an analytic CDF map.
    reference = np.resize(np.array([10, 20, 30, 40], dtype=np.uint8), (2, 4, 4))
    moving = np.stack([2 * reference + 3 * c for c in range(4)], axis=-1)
    rounds = {"round1": np.stack([reference + c for c in range(4)], axis=-1), "round4": moving}
    calls = []
    def generate(book, *, config):
        calls.append(config)
        return SimpleNamespace(rounds=rounds, channel_labels=tuple(f"ch0{i}" for i in range(4)),
            provenance=dict(image_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in rounds.items()}))
    monkeypatch.setattr("starfinder.synthetic.generate_amplicon_scene", generate)
    inputs = dict(image=_input(), reference=_input("round1", channel="ch00"))
    for entry in inputs.values():
        entry["scene"].update(condition="combined", seed=7)
    trial = dict(method=_method(recipe), inputs=inputs, seed=7, root_output_path=str(tmp_path / "out"))
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(trial))
    assert _main(path) == 0
    assert len(calls) == 1
    # The adapter is exercised with a bounded generator stub, never a full-size test image.
    assert calls[0].shape_zyx == (30, 512, 512) and calls[0].seed == 7
    expected = (reference.astype(np.int64) - 10) * 255 // 30 if recipe else reference
    result = np.load(tmp_path / "out/image.npy")
    np.testing.assert_array_equal(result, np.repeat(expected[..., None], 4, axis=-1))
    record = json.loads((tmp_path / "out/record.json").read_text())
    assert record["status"] == "succeeded"
    assert len(json.loads((tmp_path / "out/scenes.json").read_text())) == 1
    assert (tmp_path / "out/reference_preprocessing.json").exists() == recipe
    for entry in record["outputs"]:
        output = tmp_path / "out" / entry["path"]
        assert entry["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("field,value,match", [
    ("preset", "unknown", "unsupported scene preset"),
    ("version", "old", "version disagrees"),
    ("condition", "unknown", "unknown uncalibrated preprocessing condition"),
    ("seed", 2, "scene seed must equal"),
    ("seed", True, "scene seed must equal"),
    ("k", 2.0, "finite positive saturation multiplier"),
])
def test_scene_identity_rejects_invalid_or_changed_inputs(tmp_path, field, value, match):
    entry = _input()
    entry["scene"].update(condition="clean", seed=1)
    entry["scene"][field] = value
    with pytest.raises(ValueError, match=match):
        _resolve(dict(method=_method(), inputs=dict(image=entry), seed=1), tmp_path)


@pytest.mark.parametrize("field,value,match", [
    ("round", "round5", "round is not"), ("channel", "ch04", "channel is not"),
    ("scene_configuration_sha256", "changed", "SHA-256 changed"), ("path", "unused.npy", "unsupported scene input"),
])
def test_scene_selection_and_configuration_hash_are_validated(tmp_path, field, value, match):
    entry = _input()
    entry["scene"].update(condition="clean", seed=1)
    entry[field] = value
    with pytest.raises(ValueError, match=match):
        _resolve(dict(method=_method(), inputs=dict(image=entry), seed=1), tmp_path)


def test_reference_preparation_requires_an_input_and_cannot_require_another_reference(tmp_path):
    entry = _input()
    entry["scene"].update(condition="clean", seed=1)
    with pytest.raises(ValueError, match="reference_steps require a reference input"):
        _resolve(dict(method=_method(True), inputs=dict(image=entry), seed=1), tmp_path)
    method = _method(True)
    method["reference_steps"] = [dict(method="histogram_matching")]
    with pytest.raises(ValueError, match="cannot themselves require a reference"):
        _resolve(dict(method=method, inputs=dict(image=entry, reference=entry), seed=1), tmp_path)


def test_scene_trial_cleans_generated_records_after_method_failure(tmp_path, monkeypatch):
    # Reuse the bounded adapter exercise, but fail after scene generation.
    image = np.ones((2, 4, 4, 4), dtype=np.uint8)
    monkeypatch.setattr("starfinder.synthetic.generate_amplicon_scene", lambda *args, **kwargs:
        SimpleNamespace(rounds={"round4": image}, channel_labels=("ch00", "ch01", "ch02", "ch03"),
                        provenance=dict(image_sha256={"round4": "bounded-stub"})))
    def fail(*args, **kwargs):
        raise ValueError("bounded runtime failure")
    monkeypatch.setattr("starfinder.preprocessing.run_step", fail)
    entry = _input()
    entry["scene"].update(condition="clean", seed=1)
    trial = dict(method=_method(), inputs=dict(image=entry), seed=1, root_output_path=str(tmp_path / "out"))
    source = tmp_path / "trial.json"
    source.write_text(json.dumps(trial))
    assert _main(source) == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["record.json"]
