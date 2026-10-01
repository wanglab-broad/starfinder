"""Spotiflow with verified weights (W-272; docs/spot-finding-algorithms.md, "Spotiflow", checks S1, S2, S13 to
S15; docs/spot-finding-contract.md, "Pretrained weights" and "Execution device").

Extended tier, CPU only, one thread: skipped without the spotiflow extra, and failing with
MissingWeightsError when the extra is installed but the weights were not fetched into
STARFINDER_WEIGHTS_DIR. Bounds and their sources:
* S1 and S2 on the W-266 isolated-spot scenes, seeds 100-102, native defaults (W-266 detectors.csv and
  localization-per-axis.csv): recall 1.0 and precision >= 0.98 (W-266: recall 1.0, precision 0.98 to 1.0);
  3D models on iso3d 3D distance <= 0.5 voxels (W-266 max 0.432); 2D models on iso_z1 lateral <= 0.5 px
  (W-266 max 0.404).
* S13: identical table digests for two runs in one process and one in a second process, seed 100
  (W-266: two one-thread runs bit-identical).
* S15: a 3D model raises IncompatibleGeometryError on Z=1 and Z=6 and runs on 7x8x8; a 2D model raises on
  Z>1 and on 1x5x5 (W-266 minimum-shape and dimensionality probes).
* S14, provisional (a contract rule; W-266 showed only that the models load from explicit local paths):
  the weights are re-hashed before Spotiflow.from_folder is called, and a detection needs neither the
  network nor ~/.spotiflow or the Hugging Face cache.
"""
from dataclasses import replace
from functools import cache
import json
import os
import socket
import urllib.request

import numpy as np
import pytest

from starfinder.__main__ import main
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.image import IncompatibleGeometryError
from starfinder.spot_finding import KNOWN_WEIGHTS, SpotiflowConfig, WeightsHashMismatchError
from starfinder.spot_finding import _learned
from starfinder.spot_finding._weights import weights_directory

from .learned_detectors import (THREAD_VARIABLES, detect, evaluate, one_thread_environment, run_python,
    table_digest)
from .spot_finding_scenes import SEEDS, isolated_scene
from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset

pytestmark = pytest.mark.extended

MODELS_3D, MODELS_2D = ("synth_3d", "smfish_3d"), ("general", "hybiss")
SCENE = {**{m: "iso3d" for m in MODELS_3D}, **{m: "iso_z1" for m in MODELS_2D}}
# The prob_thresh stored with each model (thresholds.yaml prob_thresh_best; W-266 known-weights.csv).
STORED = {"synth_3d": 0.3, "smfish_3d": 0.4, "general": 0.49999999999999994, "hybiss": 0.5319999999999999}
READS = ("best.pt", "config.yaml", "thresholds.yaml")


@pytest.fixture(scope="module")
def torch():
    pytest.importorskip("spotiflow")
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


@cache
def isolated(model, seed):
    """(image, truth, result) of the model's isolated-spot scene; the first run of each case in this process."""
    image, truth = isolated_scene(SCENE[model], seed)
    return image, truth, detect(image, SpotiflowConfig(model))


# --- The cached weights ------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS_3D + MODELS_2D)
def test_the_cached_weights_pass_starfinder_weights_verify(model, capsys):
    assert main(["weights", "verify", "spotiflow", model]) == 0
    assert capsys.readouterr().out.startswith(f"spotiflow\t{model}\tverified\t{weights_directory()}")


# --- S1 and S2 -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS_3D)
def test_s1_s2_3d_models_on_iso3d(model, seed):
    _, truth, result = isolated(model, seed)
    match, errors = evaluate(result.spots, truth)
    assert match.values["recall"] == 1.0
    assert match.values["precision"] >= 0.98
    assert errors.values["dist_max"] <= 0.5


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS_2D)
def test_s1_s2_2d_models_on_iso_z1(model, seed):
    _, truth, result = isolated(model, seed)
    match, errors = evaluate(result.spots, truth)
    assert (result.spots.z == 0).all()
    assert match.values["recall"] == 1.0
    assert match.values["precision"] >= 0.98
    assert errors.values["lateral_max"] <= 0.5


# --- S13 -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def second_process_digests(torch):
    code = ("import json\n"
            "from starfinder.spot_finding import SpotiflowConfig\n"
            "from test.learned_detectors import detect, table_digest\n"
            "from test.spot_finding_scenes import isolated_scene\n"
            f"scenes = {SCENE!r}\n"
            "print(json.dumps({m: table_digest(detect(isolated_scene(s, 100)[0], SpotiflowConfig(m)).spots)"
            " for m, s in scenes.items()}))\n")
    return json.loads(run_python(code))


@pytest.mark.parametrize("model", MODELS_3D + MODELS_2D)
def test_s13_tables_are_identical_in_one_process_and_in_a_second(model, second_process_digests):
    image, _, first = isolated(model, 100)
    second = detect(image, SpotiflowConfig(model))
    assert len(first.spots) == 100
    assert table_digest(first.spots) == table_digest(second.spots) == second_process_digests[model]


# --- S15 -------------------------------------------------------------------------------------------------

def spot(shape):
    """W-266's minimum-shape probe image: float32, baseline 100, one Gaussian spot (sigma 1.3, amplitude 1500)
    at the centre."""
    grids = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    r2 = sum((g - (n - 1) / 2) ** 2 / 1.3 ** 2 for g, n in zip(grids, shape))
    return (100 + 1500 * np.exp(-0.5 * r2)).astype(np.float32)


@pytest.mark.parametrize("model, shape", [("smfish_3d", (1, 32, 32)), ("smfish_3d", (6, 32, 32)),
                                          ("synth_3d", (8, 7, 7)), ("general", (2, 32, 32)),
                                          ("general", (1, 5, 5)), ("hybiss", (8, 32, 32))])
def test_s15_shapes_where_spotiflow_returns_nothing_or_fails_raise(model, shape, no_loaded_models, monkeypatch):
    from spotiflow.model import Spotiflow
    calls = []
    monkeypatch.setattr(Spotiflow, "from_folder", classmethod(lambda cls, *a, **k: calls.append(a)))
    with pytest.raises(IncompatibleGeometryError):
        detect(spot(shape), SpotiflowConfig(model))
    assert calls == []


def test_s15_a_3d_model_runs_on_7x8x8_and_a_2d_model_on_a_6x6_plane():
    volume = detect(spot((7, 8, 8)), SpotiflowConfig("smfish_3d"))
    plane = detect(spot((1, 6, 6)), SpotiflowConfig("general")).spots
    assert list(volume.spots.columns) == ["spot_id", "z", "y", "x", "channel", "peak_intensity", "probability"]
    assert volume.diagnostics["geometry"] == {"n_tiles": (1, 1, 1)}
    assert len(plane) == 1 and plane.z.tolist() == [0.0]
    assert np.allclose(plane[["y", "x"]].to_numpy(), [[2.5, 2.5]], atol=0.5)


# --- S14: hash checks before the model, no network, no library caches ---------------------------------------

@pytest.mark.parametrize("changed", ["best.pt", "config.yaml", "thresholds.yaml"])
def test_s14_a_changed_hash_raises_before_spotiflow_from_folder(changed, no_loaded_models, monkeypatch):
    from spotiflow.model import Spotiflow
    calls = []
    monkeypatch.setattr(Spotiflow, "from_folder", classmethod(lambda cls, *a, **k: calls.append(a)))
    entry = KNOWN_WEIGHTS[("spotiflow", "smfish_3d")]
    wrong = lambda files: tuple(replace(f, sha256="0" * 64) if f.path == changed else f for f in files)
    monkeypatch.setitem(KNOWN_WEIGHTS, ("spotiflow", "smfish_3d"),
                        replace(entry, files=wrong(entry.files), extracted=wrong(entry.extracted)))
    with pytest.raises(WeightsHashMismatchError, match=f"{changed} has SHA-256 [0-9a-f]{{64}}, expected 0{{64}}"):
        detect(spot((8, 16, 16)), SpotiflowConfig("smfish_3d"))
    assert calls == []


def test_the_model_is_built_once_per_process_from_the_verified_folder(no_loaded_models, monkeypatch):
    from spotiflow.model import Spotiflow
    original, calls = Spotiflow.from_folder.__func__, []

    def spy(cls, *args, **kwargs):
        calls.append((args, kwargs))
        return original(cls, *args, **kwargs)
    monkeypatch.setattr(Spotiflow, "from_folder", classmethod(spy))
    image = np.stack([spot((1, 16, 16))] * 2, axis=-1)
    for _ in range(2):
        assert detect(image, SpotiflowConfig("general")).spots.channel.tolist() == [0, 1]
    assert calls == [((str(weights_directory() / "spotiflow" / "general"),), {"map_location": "cpu"})]


def test_s14_a_detection_completes_with_the_network_patched_to_raise(no_loaded_models, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a detection tried to use the network")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    image, truth = isolated_scene("iso3d", 100)
    match, _ = evaluate(detect(image, SpotiflowConfig("smfish_3d")).spots, truth)
    assert match.values["recall"] == 1.0


def test_s14_no_library_cache_is_read_or_written_with_an_empty_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = one_thread_environment(HOME=str(home), STARFINDER_WEIGHTS_DIR=str(weights_directory()))
    for name in ("XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "SPOTIFLOW_CACHE_DIR",
                 "TORCH_HOME"):
        env.pop(name, None)
    code = ("import socket, urllib.request\n"
            "def refuse(*args, **kwargs):\n"
            "    raise AssertionError('a detection tried to use the network')\n"
            "socket.socket = refuse\n"
            "urllib.request.urlopen = refuse\n"
            "from starfinder.spot_finding import SpotiflowConfig\n"
            "from test.learned_detectors import detect\n"
            "from test.spot_finding_scenes import isolated_scene\n"
            "for model, scene in (('smfish_3d', 'iso3d'), ('general', 'iso_z1')):\n"
            "    print(len(detect(isolated_scene(scene, 100)[0], SpotiflowConfig(model)).spots))\n")
    assert run_python(code, env).split() == ["100", "100"]
    assert not (home / ".spotiflow").exists() and not (home / ".cache" / "huggingface").exists()
    assert sorted(str(p.relative_to(home)) for p in home.rglob("*")) == []


# --- Records ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS_3D + MODELS_2D)
def test_records_effective_settings_model_execution_and_columns(model):
    image, _, result = isolated(model, 100)
    entry = KNOWN_WEIGHTS[("spotiflow", model)]
    diagnostics = result.diagnostics
    (effective,) = diagnostics["effective_settings"].values()
    assert effective["prob_thresh"] == STORED[model] == entry.native_threshold
    assert diagnostics["thresholds"] == (STORED[model],)
    assert (effective["subpix"], effective["scale"], effective["model"]) == (True, 1.0, model)
    assert effective["n_tiles"] == diagnostics["geometry"]["n_tiles"] == ((1, 1, 1) if model in MODELS_3D
                                                                          else (1, 1))
    folder = weights_directory() / "spotiflow" / model
    files = {f.path: f for f in entry.extracted}
    assert diagnostics["model"] == {
        "method": "spotiflow", "model": model, "training_pixel_size": entry.training_pixel_size,
        "training_pixel_size_provenance": entry.training_pixel_size_provenance,
        "artifacts": [{"name": f"spotiflow/{model}", "path": str((folder / name).resolve()),
                       "sha256": files[name].sha256, "source": entry.url, "revision": "spotiflow-models release 0.6.0"}
                      for name in READS]}
    execution = diagnostics["execution"]
    assert (execution["device"], execution["framework"]["version"], execution["framework"]["cuda"]) == (
        "cpu", "2.7.1+cpu", None)
    assert execution["threads"]["torch_num_threads"] == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert all(execution["threads"][name] == "1" for name in THREAD_VARIABLES)
    spots = result.spots
    assert list(spots.columns) == ["spot_id", "z", "y", "x", "channel", "peak_intensity", "probability"]
    assert spots.probability.between(STORED[model], 1.0).all() and spots.probability.dtype == np.float64
    index = tuple(np.clip(np.rint(spots[a].to_numpy()).astype(int), 0, n - 1) for a, n in zip("zyx", image.shape))
    assert spots.peak_intensity.tolist() == image[index].astype(float).tolist()


def test_n_tiles_is_passed_and_recorded():
    image, _ = isolated_scene("iso_z1", 100)
    result = detect(image, SpotiflowConfig("general", n_tiles=(2, 2)))
    assert result.diagnostics["geometry"] == {"n_tiles": (2, 2)}
    assert result.diagnostics["effective_settings"]["0"]["n_tiles"] == (2, 2)
    assert len(result.spots) == len(isolated("general", 100)[2].spots)


def test_the_pipeline_records_the_loaded_files_in_run_json(tmp_path):
    config = SpotiflowConfig("smfish_3d")
    checkpoints = CheckpointConfig(stages=("candidates",), directory=tmp_path / "checkpoints")
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d").run(PipelineConfig(detection=config),
                                                               checkpoints=checkpoints)
    direct = detect(fixture_image("3d"), replace(config, channel_labels=CHANNELS))
    assert fov.spot_result.spots.equals(direct.spots) and len(direct.spots) > 0
    data = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    (entry,) = [step for step in data["steps"] if step["name"] == "find_spots"][0]["methods"]
    assert entry["method"] == "spotiflow" and entry["requires"] == {"spotiflow": "0.6.5", "torch": "2.7.1+cpu"}
    assert entry["artifacts"] == direct.diagnostics["model"]["artifacts"]
    assert [a["path"].rsplit("/", 1)[1] for a in entry["artifacts"]] == list(READS)
