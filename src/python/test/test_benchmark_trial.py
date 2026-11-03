"""Private process trials and deterministic experiment expansion, without an engine."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from starfinder.benchmark._experiment import expand_experiment
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
