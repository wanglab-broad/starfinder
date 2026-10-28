"""Piscis with verified weights (W-272; design/specs/spot-finding-algorithms.md, "Piscis", checks S14 and S15;
design/specs/spot-finding.md, "Pretrained weights", "Z=1, dimensionality, scale and tiling").

Extended tier, CPU only, one thread: skipped without the piscis extra, and failing with MissingModelError
when the extra is installed but the weights were not fetched into STARFINDER_WEIGHTS_DIR. Both models run
every check; the contract does not choose between them. The S1, S2, S8 and S13 cases this file repeated, with
the same models, scenes, seeds and bounds, run only in test_spot_finding_validation.py, which holds all their
assertions (W-537). Bounds and their sources:
* S15: plane mode runs on 1x8x8 (z=0) and stack mode on 2x8x8 (W-266 minimum-shape probes).
* S14, provisional (a contract rule; W-266 showed only that the absolute-path wrapper loads the file):
  the weights are re-hashed before the Piscis constructor is called, the constructor receives an absolute
  model path even when STARFINDER_WEIGHTS_DIR is relative, and a detection needs neither the network nor
  ~/.piscis/models or the Hugging Face cache.
"""
from dataclasses import replace
from functools import cache
import json
import os
from pathlib import Path
import shutil
import socket
import urllib.request

import numpy as np
import pytest

from starfinder.__main__ import main
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.spot_finding import KNOWN_MODELS, PiscisConfig, ModelHashMismatchError
from starfinder.spot_finding import _learned
from starfinder._weights import weights_directory

from .learned_detectors import THREAD_VARIABLES, detect, evaluate, one_thread_environment, run_python
from .spot_finding_scenes import isolated_scene
from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset

pytestmark = [pytest.mark.extended, pytest.mark.spot_finding, pytest.mark.learned]

MODELS = ("20230905", "20251212")
# Stack mode on a 3D scene takes 12 s or more per detection on one CPU; plane mode on Z=1 does not.
SCENE_PARAMS = [pytest.param("iso3d", marks=pytest.mark.slow), "iso_z1"]


@pytest.fixture(scope="module")
def torch():
    pytest.importorskip("piscis")
    return pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def one_thread(torch, monkeypatch):
    """One numerical thread and no GPU for every call (the checks also set these before the process starts)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    torch.set_num_threads(1)


@pytest.fixture
def no_loaded_models(monkeypatch):
    """An empty per-process model cache, so the test's detection constructs its model."""
    monkeypatch.setattr(_learned, "_MODELS", {})


@pytest.fixture
def constructor_calls(monkeypatch):
    """A spy on the Piscis constructor, which records each call and builds the model."""
    import piscis
    original, calls = piscis.Piscis, []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(piscis, "Piscis", spy)
    return calls


@cache
def isolated(scene, model, seed):
    """(image, truth, result) of an isolated-spot scene; the first run of each case in this process."""
    image, truth = isolated_scene(scene, seed)
    return image, truth, detect(image, PiscisConfig(model))


# --- The cached weights ------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_the_cached_weights_pass_starfinder_weights_verify(model, capsys):
    assert main(["weights", "verify", "piscis", model]) == 0
    assert capsys.readouterr().out.startswith(f"piscis\t{model}\tverified\t{weights_directory()}")


# --- S15 -------------------------------------------------------------------------------------------------

def spot(shape):
    """W-266's minimum-shape probe image: float32, baseline 100, one Gaussian spot (sigma 1.3, amplitude 1500)
    at the centre."""
    grids = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    r2 = sum((g - (n - 1) / 2) ** 2 / 1.3 ** 2 for g, n in zip(grids, shape))
    return (100 + 1500 * np.exp(-0.5 * r2)).astype(np.float32)


@pytest.mark.validation
@pytest.mark.parametrize("model", MODELS)
def test_s15_plane_mode_runs_on_1x8x8_and_stack_mode_on_2x8x8(model):
    plane = detect(spot((1, 8, 8)), PiscisConfig(model))
    stack = detect(spot((2, 8, 8)), PiscisConfig(model))
    assert plane.diagnostics["geometry"]["mode"] == "plane" and stack.diagnostics["geometry"]["mode"] == "stack"
    assert len(plane.spots) == 1 and plane.spots.z.tolist() == [0.0]
    assert np.allclose(plane.spots[["y", "x"]].to_numpy(), [[3.5, 3.5]], atol=0.5)
    assert list(stack.spots.columns) == ["spot_id", "z", "y", "x", "channel", "peak_intensity"]
    assert stack.spots.z.isin([0.0, 1.0]).all()


# --- S14: hash checks before the model, no network, no library caches ---------------------------------------

@pytest.mark.validation
def test_s14_a_changed_hash_raises_before_the_piscis_constructor(no_loaded_models, constructor_calls, monkeypatch):
    entry = KNOWN_MODELS[("piscis", "20251212")]
    monkeypatch.setitem(KNOWN_MODELS, ("piscis", "20251212"),
                        replace(entry, files=(replace(entry.files[0], sha256="0" * 64),)))
    with pytest.raises(ModelHashMismatchError, match="20251212.pt has SHA-256 [0-9a-f]{64}, expected 0{64}"):
        detect(spot((2, 16, 16)), PiscisConfig("20251212"))
    assert constructor_calls == []


def test_the_model_is_built_once_per_process_from_the_absolute_path(no_loaded_models, constructor_calls):
    image = np.stack([spot((1, 16, 16))] * 2, axis=-1)
    for config in (PiscisConfig("20230905"), PiscisConfig("20230905", input_size=16)):
        spots = detect(image, config).spots
        first, second = (spots[spots.channel == c][["z", "y", "x"]].to_numpy() for c in (0, 1))
        assert len(first) > 0 and np.array_equal(first, second)
    path = weights_directory() / "piscis" / "20230905" / "20230905"
    assert constructor_calls == [((), {"model_name": str(path), "device": "cpu"})]


@pytest.mark.validation
def test_s14_a_detection_completes_with_the_network_patched_to_raise(no_loaded_models, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a detection tried to use the network")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    image, truth = isolated_scene("iso_z1", 100)
    result = detect(image, PiscisConfig("20251212"))
    match, _ = evaluate(result.spots, truth)
    assert match.values["recall"] == 1.0
    # The case test_spot_finding_validation.py's S14 ran with the same model and scene (W-537).
    assert len(result.spots) > 0 and result.diagnostics["model"]["model"] == "20251212"


@pytest.mark.validation
def test_s14_a_relative_weights_root_reaches_piscis_as_an_absolute_path(tmp_path, no_loaded_models,
                                                                         constructor_calls, monkeypatch):
    """STARFINDER_WEIGHTS_DIR=weights, relative to a working directory that holds a copy of the cache; Piscis
    would otherwise prefix the relative name with its MODELS_DIR (~/.piscis/models) and could download."""
    work, home = tmp_path / "work", tmp_path / "home"
    home.mkdir()
    shutil.copytree(weights_directory() / "piscis" / "20251212", work / "weights" / "piscis" / "20251212")

    def refuse(*args, **kwargs):
        raise AssertionError("a detection tried to use the network")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", "weights")
    image, truth = isolated_scene("iso_z1", 100)
    result = detect(image, PiscisConfig("20251212"))
    path = work.resolve() / "weights" / "piscis" / "20251212" / "20251212"
    assert constructor_calls == [((), {"model_name": str(path), "device": "cpu"})]
    assert Path(constructor_calls[0][1]["model_name"]).is_absolute()
    assert [a["path"] for a in result.diagnostics["model"]["artifacts"]] == [f"{path}.pt"]
    assert evaluate(result.spots, truth)[0].values["recall"] == 1.0
    assert list(home.iterdir()) == []


@pytest.mark.validation
@pytest.mark.slow
def test_s14_no_library_cache_is_read_or_written_with_an_empty_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = one_thread_environment(HOME=str(home), STARFINDER_WEIGHTS_DIR=str(weights_directory()))
    for name in ("XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        env.pop(name, None)
    code = ("import socket, urllib.request\n"
            "def refuse(*args, **kwargs):\n"
            "    raise AssertionError('a detection tried to use the network')\n"
            "socket.socket = refuse\n"
            "urllib.request.urlopen = refuse\n"
            "import piscis.paths\n"
            "print(piscis.paths.MODELS_DIR)\n"
            "from starfinder.spot_finding import PiscisConfig\n"
            "from test.learned_detectors import detect\n"
            "from test.spot_finding_scenes import isolated_scene\n"
            "for model in ('20230905', '20251212'):\n"
            "    print(len(detect(isolated_scene('iso_z1', 100)[0], PiscisConfig(model)).spots))\n")
    assert run_python(code, env).split() == [str(home / ".piscis" / "models"), "100", "100"]
    assert not (home / ".piscis").exists() and not (home / ".cache" / "huggingface").exists()
    assert sorted(str(p.relative_to(home)) for p in home.rglob("*")) == []


# --- Records ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("scene", SCENE_PARAMS)
def test_records_effective_settings_model_execution_and_columns(scene, model):
    image, _, result = isolated(scene, model, 100)
    entry = KNOWN_MODELS[("piscis", model)]
    diagnostics = result.diagnostics
    (effective,) = diagnostics["effective_settings"].values()
    assert (effective["threshold"], effective["min_distance"], effective["input_size"], effective["scale"],
            effective["model"]) == (0.5, 1, 256, 1.0, model)
    assert diagnostics["thresholds"] == (0.5,) and entry.native_threshold == 0.5
    assert diagnostics["geometry"]["mode"] == ("stack" if scene == "iso3d" else "plane")
    path = weights_directory() / "piscis" / model / f"{model}.pt"
    assert diagnostics["model"] == {
        "method": "piscis", "model": model, "training_pixel_size": "not published by the authors",
        "training_pixel_size_provenance": entry.training_pixel_size_provenance,
        "artifacts": [{"name": f"piscis/{model}", "path": str(path.resolve()), "sha256": entry.sha256,
                       "source": entry.url, "revision": "wniu/Piscis 9bdefc72cb519053c63fd2d7bff9d12db7bb394e"}]}
    execution = diagnostics["execution"]
    assert (execution["device"], execution["framework"]["version"], execution["framework"]["cuda"]) == (
        "cpu", "2.7.1+cpu", None)
    assert execution["threads"]["torch_num_threads"] == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert all(execution["threads"][name] == "1" for name in THREAD_VARIABLES)
    spots = result.spots
    assert list(spots.columns) == ["spot_id", "z", "y", "x", "channel", "peak_intensity"]
    index = tuple(np.clip(np.rint(spots[a].to_numpy()).astype(int), 0, n - 1) for a, n in zip("zyx", image.shape))
    assert spots.peak_intensity.tolist() == image[index].astype(float).tolist()


def test_the_pipeline_records_the_loaded_file_in_run_json(tmp_path):
    config = PiscisConfig("20251212", input_size=48)
    checkpoints = CheckpointConfig(stages=("candidates",), directory=tmp_path / "checkpoints")
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d").run(PipelineConfig(spot_finding=config),
                                                               checkpoints=checkpoints)
    direct = detect(fixture_image("3d"), replace(config, channel_labels=CHANNELS))
    assert fov.spot_result.spots.equals(direct.spots) and len(direct.spots) > 0
    data = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    (entry,) = [step for step in data["steps"] if step["name"] == "find_spots"][0]["methods"]
    assert entry["method"] == "piscis" and entry["requires"] == {"piscis": "1.1.0", "torch": "2.7.1+cpu"}
    assert entry["artifacts"] == direct.diagnostics["model"]["artifacts"]
    assert entry["artifacts"][0]["sha256"] == KNOWN_MODELS[("piscis", "20251212")].sha256
