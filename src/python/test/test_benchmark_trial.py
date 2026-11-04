"""Private process trials and deterministic experiment expansion, without an engine."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder._records import jsonable
from starfinder.benchmark._experiment import expand_experiment
from starfinder.benchmark._trial import _main, _resolve
from starfinder.synthetic import UNCALIBRATED_PREPROCESSING_VERSION
from starfinder.workflow._shape import configuration_sha256

pytestmark = [pytest.mark.benchmark, pytest.mark.contract]


def _inputs(tmp_path):
    z, y, x = np.indices((8, 24, 24))
    image = (100 * np.exp(-((z - 3)**2 / 2 + (y - 9)**2 / 8 + (x - 10)**2 / 8))
             + 60 * np.exp(-((z - 5)**2 / 2 + (y - 17)**2 / 5 + (x - 17)**2 / 5))).astype(np.float32)
    np.save(tmp_path / "reference.npy", image)
    np.save(tmp_path / "moving.npy", np.roll(image, 1, axis=2))
    return image


def _run(tmp_path, value):
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, "-m", "starfinder.benchmark._trial", str(path)],
                            capture_output=True, text=True, timeout=120)
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    return result, record


@pytest.mark.parametrize("module,method,config", [
    ("registration", "translation", {}),
    pytest.param("registration", "bspline", dict(iterations=1, levels=1, samples=128,
                                                  grid_spacing_physical=8), marks=pytest.mark.slow),
    ("registration", "demons", dict(iterations=[2], pyramid_mode="sitk")),
    ("preprocessing", "min_max_normalization", dict(output_dtype="uint8", output_range=[0, 255])),
])
def test_private_trial_subprocess(tmp_path, module, method, config):
    if method in ("demons", "bspline") and importlib.util.find_spec("SimpleITK") is None:
        pytest.skip("SimpleITK required for this execution coverage")
    if method == "bspline" and importlib.util.find_spec("itk") is None:
        pytest.skip("itk-elastix required for B-spline execution coverage")
    image = _inputs(tmp_path)
    entry = dict(module=module, method=method, config=config) if module == "registration" else dict(
        module=module, steps=[dict(method=method, config=config),
                              dict(method="white_tophat", config=dict(radius_yx=1))])
    inputs = dict(reference="reference.npy", moving="moving.npy") if module == "registration" else dict(image="reference.npy")
    result, record = _run(tmp_path, dict(method=entry, inputs=inputs, seed=17, root_output_path="out"))
    assert result.returncode == 0, (result.stdout, result.stderr, record)
    assert record["status"] == "succeeded" and record["error"] is None
    assert record["wall_s"] > 0 and record["cpu_s"] > 0 and record["peak_rss_kib"] > 0
    assert record["seed"] == 17
    assert record["configuration_sha256"] == configuration_sha256(record["configuration"])
    for role, item in record["inputs"].items():
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
    for output in record["outputs"]:
        path = tmp_path / "out" / output["path"]
        assert output["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert output["bytes"] == path.stat().st_size
    output = np.load(tmp_path / "out" / "image.npy")
    assert output.shape == image.shape and np.isfinite(output).all()
    if module == "registration":
        transform = json.loads((tmp_path / "out" / "transform.json").read_text())
        assert transform["diagnostics"]["effective_config"]["method"] == method
        if method in ("demons", "bspline"):
            with np.load(tmp_path / "out" / "transform.npz") as arrays:
                assert np.isfinite(arrays["result_0"]).all()
        else:
            np.testing.assert_allclose(transform["transform"]["displacement_zyx"], [0, 0, 1], rtol=1e-9, atol=1e-12)
    else:
        assert output.dtype == np.uint8
        assert json.loads((tmp_path / "out" / "preprocessing.json").read_text())[0]["config"]["type"] == "MinMaxNormalizationConfig"


@pytest.mark.parametrize("failure", ["method", "input", "runtime"])
def test_private_trial_failure_record(tmp_path, failure):
    _inputs(tmp_path)
    method = "unknown" if failure == "method" else "translation"
    inputs = dict(reference="reference.npy", moving="moving.npy")
    if failure == "input":
        inputs["moving"] = dict(path="moving.npy", sha256="wrong")
    if failure == "runtime":
        np.save(tmp_path / "moving.npy", np.full((8, 24, 24), np.nan))
    result, record = _run(tmp_path, dict(method=dict(module="registration", method=method), inputs=inputs,
                                        seed=17, root_output_path="out"))
    assert result.returncode == 1
    assert record["status"] == "failed" and record["error"]["type"]
    assert record["error"]["message"] and record["outputs"] == []
    assert record["wall_s"] > 0 and record["peak_rss_kib"] > 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["record.json"]


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_experiment_expands_product_and_stable_manifest(tmp_path, suffix):
    _inputs(tmp_path)
    experiment = dict(methods=[dict(module="registration", method="translation"),
                               dict(module="registration", method="demons", config=dict(iterations=[2]))],
                      conditions=[dict(name="first"), dict(name="second")], seeds=[7, 11],
                      inputs=[dict(reference="reference.npy", moving="moving.npy"),
                              dict(reference="moving.npy", moving="reference.npy")])
    path = tmp_path / f"experiment.{suffix}"
    path.write_text(json.dumps(experiment) if suffix == "json" else yaml.safe_dump(experiment))
    output = tmp_path / "expanded"
    first = expand_experiment(path, output)
    before = {p.name: p.read_bytes() for p in output.glob("*.json")}
    second = expand_experiment(path, output)
    assert first == second == json.loads((output / "manifest.json").read_text())
    assert before == {p.name: p.read_bytes() for p in output.glob("*.json")}
    assert len(first["trials"]) == 16
    assert len({entry["configuration_sha256"] for entry in first["trials"]}) == 16
    for entry in first["trials"]:
        trial = output / entry["path"]
        assert entry["sha256"] == "sha256:" + hashlib.sha256(trial.read_bytes()).hexdigest()
        configuration = json.loads(trial.read_text())
        assert entry["configuration_sha256"] == configuration_sha256(configuration)
        assert configuration["method"]["config"]["type"].endswith("Config")
    # The expanded files are executable, not just descriptions.
    result = subprocess.run([sys.executable, "-m", "starfinder.benchmark._trial", str(output / first["trials"][0]["path"])],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "trial-000000" / "record.json").read_text())["configuration_sha256"] == first["trials"][0]["configuration_sha256"]


def test_experiment_condition_resolves_overrides_and_relocation(tmp_path):
    _inputs(tmp_path)
    experiment = dict(methods=[dict(module="registration", method="translation", config=dict(fft_workers=1))],
                      conditions=[dict(name="scipy"),
                                  dict(name="skimage", method=dict(config=dict(backend="skimage")))],
                      seeds=[7], inputs=[dict(reference="reference.npy", moving="moving.npy")])
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    first = expand_experiment(path, tmp_path / "first")
    second = expand_experiment(path, tmp_path / "second")
    assert [e["configuration_sha256"] for e in first["trials"]] == [e["configuration_sha256"] for e in second["trials"]]
    trials = [json.loads((tmp_path / "first" / e["path"]).read_text()) for e in first["trials"]]
    assert [t["method"]["config"]["backend"] for t in trials] == ["scipy_fft", "skimage"]
    assert all(t["method"]["config"]["fft_workers"] == 1 for t in trials)


# Spot-finding and segmentation trials (W-797): bounded fixtures, the stage entries called directly for equality.
_SEEDED = dict(module="segmentation", method="seeded_watershed", target="cell", roles=["cytoplasm"])
_SPACING = dict(frame_id="trial/reference", spacing_zyx=[1.0, 0.5, 0.5])


def _seeds(tmp_path, image):
    seeds = np.zeros(image.shape[:3], dtype=np.uint32)
    seeds[3, 9, 10], seeds[5, 17, 17] = 1, 2
    np.save(tmp_path / "seeds.npy", seeds)
    return seeds


def _hashed_outputs(out, record, names):
    assert [entry["path"] for entry in record["outputs"]] == names
    for entry in record["outputs"]:
        assert entry["sha256"] == hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest()
        assert entry["bytes"] == (out / entry["path"]).stat().st_size
    assert sorted(p.name for p in out.iterdir()) == sorted(names + ["record.json"])


def _direct_spots(image, config, metadata):
    from starfinder.image import ImageMetadata
    from starfinder.spot_finding import find_spots
    return find_spots(image, config=config, metadata=ImageMetadata(**metadata), spot_namespace="trial")


def _direct_labels(image, config, metadata, seeds=None):
    from starfinder.image import ImageMetadata
    from starfinder.segmentation import ReferenceGrid, SegmentationInput, SegmentationResult, segment
    image = image[..., None] if image.ndim == 3 else image
    grid = ReferenceGrid(image.shape[:3], ImageMetadata(**metadata), "declared")
    if seeds is not None:
        seeds = SegmentationResult(seeds, grid, "nucleus", "volume", "trial", {}, {})
    return segment(SegmentationInput(image, grid, ("cytoplasm",)), config=config, target="cell", seeds=seeds,
                   label_namespace="trial").labels


@pytest.mark.spot_finding
@pytest.mark.parametrize("channels", [False, True])
def test_spot_finding_trial_writes_find_spots_table(tmp_path, channels):
    from starfinder.spot_finding import LocalMaximaConfig
    image = _inputs(tmp_path)
    if channels:
        image = np.stack([image, np.roll(image, 3, axis=1)], axis=-1)
        np.save(tmp_path / "reference.npy", image)
    config = dict(threshold_mode="absolute", threshold_value=20.0, merge_radius_zyx=[1.0, 2.0, 2.0])
    result, record = _run(tmp_path, dict(method=dict(module="spot_finding", method="local_maxima", config=config),
                                         inputs=dict(image="reference.npy"), seed=17, root_output_path="out"))
    assert result.returncode == 0, (result.stdout, result.stderr, record)
    assert record["status"] == "succeeded" and record["execution"] == {"device": "cpu", "model": None}
    assert record["configuration"]["method"]["config"]["type"] == "LocalMaximaConfig"
    assert record["configuration_sha256"] == configuration_sha256(record["configuration"])
    _hashed_outputs(tmp_path / "out", record, ["spots.csv", "spot_finding.json"])
    direct = _direct_spots(image, LocalMaximaConfig(**dict(config, merge_radius_zyx=(1.0, 2.0, 2.0))),
                           dict(frame_id="trial/reference"))
    assert len(direct.spots) >= 2
    assert (tmp_path / "out/spots.csv").read_text() == direct.spots.to_csv(index=False)
    table = pd.read_csv(tmp_path / "out/spots.csv")
    for column in ("z", "y", "x"):
        np.testing.assert_array_equal(table[column].to_numpy(), direct.spots[column].to_numpy())
    diagnostics = json.loads((tmp_path / "out/spot_finding.json").read_text())
    assert diagnostics["counts"] == json.loads(json.dumps(jsonable(direct.diagnostics)))["counts"]


@pytest.mark.segmentation
def test_segmentation_trial_writes_segment_labels(tmp_path):
    from starfinder.segmentation import SeededWatershedConfig
    image = _inputs(tmp_path)
    seeds = _seeds(tmp_path, image)
    result, record = _run(tmp_path, dict(method=dict(_SEEDED, config=dict(sigma_um=0.5)),
                                         inputs=dict(image="reference.npy", seeds="seeds.npy"), metadata=_SPACING,
                                         seed=17, root_output_path="out"))
    assert result.returncode == 0, (result.stdout, result.stderr, record)
    assert record["status"] == "succeeded" and record["execution"] == {"device": "cpu", "model": None}
    assert set(record["inputs"]) == {"image", "seeds"}
    _hashed_outputs(tmp_path / "out", record, ["labels.npy", "segmentation.json"])
    labels = np.load(tmp_path / "out/labels.npy")
    direct = _direct_labels(image, SeededWatershedConfig(sigma_um=0.5), _SPACING, seeds)
    assert labels.dtype == np.uint32 and set(np.unique(labels)) == {0, 1, 2}
    np.testing.assert_array_equal(labels, direct)
    saved = json.loads((tmp_path / "out/segmentation.json").read_text())
    assert saved["record"]["methods"][0]["method"] == "seeded_watershed"
    assert saved["record"]["outcome"] == "ok" and "wall_seconds" in saved["diagnostics"]


def _amplicon_stub(monkeypatch, rounds):
    def generate(book, *, config):
        return SimpleNamespace(rounds=rounds, channel_labels=tuple(f"ch0{i}" for i in range(4)),
                               provenance=dict(image_sha256={k: "bounded-stub" for k in rounds}))
    monkeypatch.setattr("starfinder.synthetic.generate_amplicon_scene", generate)


def _cellular_stub(monkeypatch, image, calls):
    def generate(book, *, config):
        calls.append(config)
        return SimpleNamespace(stains={"FOV_001": {"PI": image}}, morphology={})
    monkeypatch.setattr("starfinder.synthetic.generate_cellular_scene", generate)


def _main_record(tmp_path, trial):
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(dict(trial, root_output_path=str(tmp_path / "out"))))
    status = _main(path)
    return status, json.loads((tmp_path / "out/record.json").read_text())


@pytest.mark.spot_finding
def test_spot_finding_trial_from_scene_selects_round_and_channel(tmp_path, monkeypatch):
    # A bounded generator stub stands in for the 30 x 512 x 512 uncalibrated scene.
    from starfinder.spot_finding import LocalMaximaConfig
    image = _inputs(tmp_path)
    rounds = {"round1": np.stack([image * (c + 1) for c in range(4)], axis=-1)}
    _amplicon_stub(monkeypatch, rounds)
    scene = dict(scene=dict(preset="uncalibrated_preprocessing", version=UNCALIBRATED_PREPROCESSING_VERSION,
                            condition="clean", seed=17), round="round1", channel="ch02")
    status, record = _main_record(tmp_path, dict(method=dict(module="spot_finding", method="local_maxima"),
                                                 inputs=dict(image=scene), seed=17))
    assert status == 0 and record["status"] == "succeeded", record["error"]
    assert record["inputs"]["image"]["scene_configuration_sha256"].startswith("sha256:")
    _hashed_outputs(tmp_path / "out", record, ["scenes.json", "spots.csv", "spot_finding.json"])
    direct = _direct_spots(rounds["round1"][..., 2], LocalMaximaConfig(), dict(frame_id="trial/reference"))
    assert (tmp_path / "out/spots.csv").read_text() == direct.spots.to_csv(index=False)


@pytest.mark.segmentation
def test_segmentation_trial_from_cellular_scene_stacks_stains(tmp_path, monkeypatch):
    from starfinder.segmentation import SeededWatershedConfig
    from starfinder.synthetic import CELLULAR_PRESET_VERSION
    image = (_inputs(tmp_path) * 2).astype(np.uint8)
    seeds = _seeds(tmp_path, image)
    calls = []
    _cellular_stub(monkeypatch, image, calls)
    scene = dict(scene=dict(preset="context-tissue2d-v1", version=CELLULAR_PRESET_VERSION, condition="clean"),
                 fov="FOV_001", stains=["PI"])
    status, record = _main_record(tmp_path, dict(method=_SEEDED, inputs=dict(image=scene, seeds="seeds.npy"),
                                                 metadata=_SPACING, seed=17))
    assert status == 0 and record["status"] == "succeeded", record["error"]
    assert len(calls) == 1 and calls[0].sample.fov.seed == 17
    assert record["inputs"]["image"]["scene"]["seed"] == 17 and record["inputs"]["image"]["round"] == "reference"
    _hashed_outputs(tmp_path / "out", record, ["scenes.json", "labels.npy", "segmentation.json"])
    scenes = json.loads((tmp_path / "out/scenes.json").read_text())
    assert list(scenes[0]["image_sha256"]) == ["FOV_001/reference/PI"]
    np.testing.assert_array_equal(np.load(tmp_path / "out/labels.npy"),
                                  _direct_labels(image, SeededWatershedConfig(), _SPACING, seeds))


@pytest.mark.segmentation
@pytest.mark.parametrize("field,value,match", [
    ("fov", "FOV_009", "fov is not in the preset"), ("round", "round1", "must be 'reference' or 'morphology'"),
    ("stains", ["DAPI"], "stains must name"), ("stains", "PI", "stains must name"),
    ("channel", "ch00", "unsupported scene input keys"), ("scene_configuration_sha256", "changed", "SHA-256 changed"),
])
def test_cellular_scene_input_is_validated(tmp_path, field, value, match):
    from starfinder.synthetic import CELLULAR_PRESET_VERSION
    entry = dict(scene=dict(preset="context-tissue2d-v1", version=CELLULAR_PRESET_VERSION), fov="FOV_001",
                 stains=["PI"])
    entry[field] = value
    with pytest.raises(ValueError, match=match):
        _resolve(dict(method=_SEEDED, inputs=dict(image=entry), seed=1), tmp_path)


def test_scene_presets_belong_to_their_modules(tmp_path):
    from starfinder.synthetic import CELLULAR_PRESET_VERSION
    cellular = dict(scene=dict(preset="context-tissue2d-v1", version=CELLULAR_PRESET_VERSION), fov="FOV_001",
                    stains=["PI"])
    amplicon = dict(scene=dict(preset="uncalibrated_preprocessing", version=UNCALIBRATED_PREPROCESSING_VERSION,
                               condition="clean"), round="round1")
    with pytest.raises(ValueError, match="unsupported scene preset for spot_finding"):
        _resolve(dict(method=dict(module="spot_finding", method="local_maxima"), inputs=dict(image=cellular),
                      seed=1), tmp_path)
    with pytest.raises(ValueError, match="unsupported scene preset for segmentation"):
        _resolve(dict(method=_SEEDED, inputs=dict(image=amplicon), seed=1), tmp_path)
    with pytest.raises(ValueError, match="scene seed must equal"):
        _resolve(dict(method=_SEEDED, inputs=dict(image=dict(cellular, scene=dict(cellular["scene"], seed=2))),
                      seed=1), tmp_path)


_FAILURES = {
    ("spot_finding", "method"): (dict(module="spot_finding", method="unknown"), "ValueError", "unknown method"),
    ("spot_finding", "type"): (dict(module="spot_finding", method="local_maxima",
                                    config=dict(type="StarfishLogConfig")), "ValueError", "type disagrees"),
    ("spot_finding", "input"): (dict(module="spot_finding", method="local_maxima"), "ValueError", "inputs must name"),
    ("spot_finding", "weights"): (dict(module="spot_finding", method="spotiflow", config=dict(model="synth_3d")),
                                  "MissingModelError", "is missing"),
    ("segmentation", "method"): (dict(_SEEDED, method="unknown"), "ValueError", "unknown method"),
    ("segmentation", "type"): (dict(_SEEDED, config=dict(type="CellposeConfig")), "ValueError", "type disagrees"),
    ("segmentation", "input"): (_SEEDED, "ValueError", "unsupported input keys"),
    ("segmentation", "weights"): (dict(_SEEDED, method="stardist", target="nucleus", roles=["nuclear"],
                                       config=dict(model="2D_versatile_fluo", scale=1.0)),
                                  "MissingModelError", "is missing"),
}


@pytest.mark.spot_finding
@pytest.mark.segmentation
@pytest.mark.parametrize("module,failure", list(_FAILURES))
def test_new_module_failures_are_named_before_any_output(tmp_path, monkeypatch, module, failure):
    method, error, match = _FAILURES[module, failure]
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(tmp_path / "empty-weights"))
    def forbidden(*args, **kwargs):
        pytest.fail("A named failure must come before any input is loaded or generated")
    monkeypatch.setattr("starfinder.synthetic.generate_amplicon_scene", forbidden)
    monkeypatch.setattr("starfinder.synthetic.generate_cellular_scene", forbidden)
    _inputs(tmp_path)
    inputs = dict(image="reference.npy")
    if failure == "input":
        inputs = (dict(image="reference.npy", reference="moving.npy") if module == "spot_finding"
                  else dict(image=dict(path="reference.npy", channel="ch00")))
    if failure == "weights" and module == "spot_finding":
        inputs = dict(image=dict(scene=dict(preset="uncalibrated_preprocessing", condition="clean",
                                            version=UNCALIBRATED_PREPROCESSING_VERSION), round="round1"))
    status, record = _main_record(tmp_path, dict(method=method, inputs=inputs, seed=17))
    assert status == 1 and record["status"] == "failed" and record["outputs"] == []
    assert record["error"]["type"] == error and match in record["error"]["message"]
    assert "execution" not in record
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["record.json"]


def test_registration_records_keep_their_fields(tmp_path):
    _inputs(tmp_path)
    status, record = _main_record(tmp_path, dict(method=dict(module="registration", method="translation"),
                                                 inputs=dict(reference="reference.npy", moving="moving.npy"), seed=1))
    assert status == 0
    assert set(record) == {"format", "configuration", "seed", "inputs", "outputs", "status", "error",
                           "configuration_sha256", "wall_s", "cpu_s", "peak_rss_kib"}


@pytest.mark.segmentation
def test_segmentation_trial_generates_a_real_cellular_scene(tmp_path):
    # The smallest preset, 2 x 2 FOVs of 8 x 64 x 64, generated once inside the trial.
    from starfinder.synthetic import CELLULAR_PRESET_VERSION
    seeds = np.zeros((8, 64, 64), dtype=np.uint32)
    seeds[4, 20, 20], seeds[4, 44, 44] = 1, 2
    np.save(tmp_path / "seeds.npy", seeds)
    scene = dict(scene=dict(preset="context-tissue2d-v1", version=CELLULAR_PRESET_VERSION), fov="FOV_001",
                 stains=["PI"])
    status, record = _main_record(tmp_path, dict(method=_SEEDED, inputs=dict(image=scene, seeds="seeds.npy"),
                                                 metadata=_SPACING, seed=3))
    assert status == 0, record["error"]
    labels = np.load(tmp_path / "out/labels.npy")
    assert labels.shape == (8, 64, 64) and labels[4, 20, 20] == 1 and labels[4, 44, 44] == 2
    scenes = json.loads((tmp_path / "out/scenes.json").read_text())
    assert {"FOV_001/reference/PI", "FOV_004/reference/PI"} <= set(scenes[0]["image_sha256"])


# --- The saved trial outputs read by `starfinder experiment evaluate` (W-793; the replacements that
# design/baselines/benchmark.md names for the old lifecycle's saved-artifact tests) --------------------------------

def _evaluate_trial(tmp_path, monkeypatch, trial, metric, files=None):
    """``starfinder experiment evaluate`` of one metric on one trial output, with processing patched to raise."""
    from starfinder.__main__ import main
    from .test_experiment_evaluate import _no_processing, _tree
    sources = [dict(name="trial", kind="trial", path=str(trial))]
    if files:
        sources.append(dict(name="truth", kind="files", files={
            key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key, path in files.items()}))
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(dict(sources=sources)))
    (tmp_path / "metrics.yaml").write_text(yaml.safe_dump(dict(metrics=[metric])))
    before = _tree(trial, *(files or {}).values())
    with monkeypatch.context() as patch:
        _no_processing(patch)
        status = main(["experiment", "evaluate", "--sources", str(tmp_path / "sources.yaml"),
                       "--metrics", str(tmp_path / "metrics.yaml"), "--output", str(tmp_path / "evaluation")])
    assert _tree(trial, *(files or {}).values()) == before
    return status, json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())["metrics"][0]


def test_nonempty_output_is_preserved(tmp_path, capsys):
    """The private trial refuses a nonempty output folder, through the entry and the command; nothing in it
    changes (the old lifecycle's resume protocol is not part of a trial)."""
    from starfinder.__main__ import main
    _inputs(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "earlier.txt").write_text("an earlier attempt")
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(dict(method=dict(module="registration", method="translation"),
                                    inputs=dict(reference="reference.npy", moving="moving.npy"), seed=1,
                                    root_output_path="out")))
    with pytest.raises(FileExistsError, match="must be empty"):
        _main(path)
    with pytest.raises(SystemExit) as error:
        main(["trial", "run", "--config", str(path)])
    assert error.value.code == 2 and "must be empty" in capsys.readouterr().err
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["earlier.txt"]
    assert (tmp_path / "out" / "earlier.txt").read_text() == "an earlier attempt"


_FRACTIONAL = dict(frame_id="trial/fractional", spacing_zyx=[0.35, 0.65, 0.65], origin_zyx=[0.25, -1.5, 2.125])


def _demons(tmp_path):
    if importlib.util.find_spec("SimpleITK") is None:
        pytest.skip("SimpleITK required for the demons trial")
    _inputs(tmp_path)
    status, record = _main_record(tmp_path, dict(
        method=dict(module="registration", method="demons", config=dict(iterations=[2], pyramid_mode="sitk")),
        inputs=dict(reference="reference.npy", moving="moving.npy"), metadata=_FRACTIONAL, seed=5))
    assert status == 0, record["error"]
    return record


def test_float_transform_and_dense_artifacts(tmp_path, monkeypatch):
    """Fractional metadata survives the trial record and transform.json exactly; the dense field in transform.npz is
    the array the evaluation reads (its error against itself is exactly zero)."""
    record = _demons(tmp_path)
    assert record["configuration"]["metadata"]["spacing_zyx"] == _FRACTIONAL["spacing_zyx"]
    assert record["configuration"]["metadata"]["origin_zyx"] == _FRACTIONAL["origin_zyx"]
    transform = json.loads((tmp_path / "out" / "transform.json").read_text())
    for key in ("reference_metadata", "moving_metadata"):
        assert transform["transform"][key]["spacing_zyx"] == _FRACTIONAL["spacing_zyx"]
        assert transform["transform"][key]["origin_zyx"] == _FRACTIONAL["origin_zyx"]
    listed = {o["path"]: o for o in record["outputs"]}
    assert set(listed) == {"transform.npz", "transform.json", "image.npy"}
    npz = tmp_path / "out" / "transform.npz"
    assert listed["transform.npz"]["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    with np.load(npz) as arrays:
        field = arrays["result_0"]
    assert field.shape == (8, 24, 24, 3) and field.dtype.kind == "f" and np.isfinite(field).all()
    saved = {"source": "trial", "output": "transform.npz", "key": "result_0"}
    status, metric = _evaluate_trial(tmp_path, monkeypatch, tmp_path / "out", dict(
        name="identity", function="registration.evaluate_displacement_field",
        parameters=dict(mask=None, spacing_zyx=_FRACTIONAL["spacing_zyx"]),
        arguments=dict(estimated=saved, truth=saved)))
    assert status == 0 and metric["outcome"] == "computed"
    assert all(value == 0.0 for value in metric["values"].values())
    assert metric["counts"] == {"total": 8 * 24 * 24, "masked": 8 * 24 * 24}


def test_demons_saved_field(tmp_path, monkeypatch):
    """The saved demons field, evaluated against a supplied zero field: the error norms are the field's own norms,
    computed here independently, in voxels and in the fractional spacing's physical units."""
    _demons(tmp_path)
    with np.load(tmp_path / "out" / "transform.npz") as arrays:
        field = arrays["result_0"].astype(np.float64)
    np.save(tmp_path / "zero.npy", np.zeros_like(field))
    status, metric = _evaluate_trial(
        tmp_path, monkeypatch, tmp_path / "out", dict(
            name="demons", metric="M21", function="registration.evaluate_displacement_field",
            parameters=dict(mask=None, spacing_zyx=_FRACTIONAL["spacing_zyx"]),
            arguments=dict(estimated={"source": "trial", "output": "transform.npz", "key": "result_0"},
                           truth={"source": "truth", "file": "zero"})),
        files={"zero": tmp_path / "zero.npy"})
    assert status == 0 and metric["outcome"] == "computed"
    norms = np.linalg.norm(field, axis=-1).ravel()
    physical = np.linalg.norm(field * np.asarray(_FRACTIONAL["spacing_zyx"]), axis=-1).ravel()
    assert norms.max() > 0
    expected = {"median_error": np.median(norms), "p95_error": np.percentile(norms, 95), "max_error": norms.max(),
                "median_error_physical": np.median(physical), "p95_error_physical": np.percentile(physical, 95),
                "max_error_physical": physical.max()}
    for key, value in expected.items():
        assert metric["values"][key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


@pytest.mark.synthetic
def test_generated_registration_pair_evaluate_only(tmp_path, monkeypatch):
    """A generated registration pair (``starfinder synthetic generate --mode registration --preset tiny``), its
    translation trial, then evaluate-only: the saved displacement against the generator's truth shift, within the old
    lifecycle test's tolerance of one voxel per axis."""
    from starfinder.__main__ import main
    assert main(["synthetic", "generate", "--mode", "registration", "--preset", "tiny", "--seed", "42",
                 "--owner", "pytest", "--output", str(tmp_path / "pair")]) == 0
    root = tmp_path / "pair" / "synthetic" / "tiny"
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(dict(method=dict(module="registration", method="translation"),
                                    inputs=dict(reference=str(root / "ref.tif"), moving=str(root / "mov_shift.tif")),
                                    seed=42, root_output_path="out")))
    assert main(["trial", "run", "--config", str(path)]) == 0
    truth = json.loads((root / "ground_truth.json").read_text())["pairs"]["shift"]["shift_zyx"]
    status, metric = _evaluate_trial(
        tmp_path, monkeypatch, tmp_path / "out", dict(
            name="shift", function="registration.evaluate_translation",
            parameters=dict(units="voxel", tolerance=1.0),
            arguments=dict(
                detected={"mapping": {"moving": {"source": "trial", "output": "transform.json",
                                                 "key": "transform.displacement_zyx"}}},
                truth={"mapping": {"moving": {"source": "truth", "file": "truth", "key": "pairs.shift.shift_zyx"}}},
                reference_metadata={"source": "trial", "metadata": True},
                observed_metadata={"source": "trial", "metadata": True})),
        files={"truth": root / "ground_truth.json"})
    assert status == 0 and metric["outcome"] == "computed" and metric["values"]["passed"] is True
    displacement = json.loads((tmp_path / "out" / "transform.json").read_text())["transform"]["displacement_zyx"]
    assert metric["values"]["max_error"] == pytest.approx(
        float(np.max(np.abs(np.subtract(displacement, truth)))), rel=1e-9, abs=1e-12)
