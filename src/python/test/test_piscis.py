"""Piscis with verified weights (W-272; docs/spot-finding-algorithms.md, "Piscis", checks S1, S2, S8, S13 to
S15; docs/spot-finding-contract.md, "Pretrained weights", "Z=1, dimensionality, scale and tiling").

Extended tier, CPU only, one thread: skipped without the piscis extra, and failing with MissingWeightsError
when the extra is installed but the weights were not fetched into STARFINDER_WEIGHTS_DIR. Both models run
every check; the contract does not choose between them. Bounds and their sources:
* S1 and S2 on the W-266 isolated-spot scenes, seeds 100-102, native defaults (W-266 detectors.csv and
  localization-per-axis.csv): recall 1.0 and precision >= 0.98 (W-266: recall 1.0, precision 0.98 to 1.0 in
  3D and 1.0 on Z=1); plane mode on iso_z1 lateral <= 0.15 px (W-266 max 0.098); stack mode on iso3d
  lateral <= 0.15 px (W-266 max 0.119) and absolute Z <= 2.0 voxels, provisional (W-266 max 1.763 on three
  seeds of one scene; W-266 withdrew 1.0).
* S8 on the W-266 seam scenes restricted to the spots 0.5, 1 and 2 px from a keep-boundary and the controls,
  input_size=32 (keep-boundaries 30.5 and 59.5): exactly one candidate within 3 voxels of each spot, and a
  lateral shift <= 0.2 px against the untiled run (W-266 piscis-seams.csv: all off-boundary and control
  spots single, maximum shift 0.147 px).
* S13: identical table digests for two runs in one process and one in a second process, seed 100
  (W-266: two one-thread runs bit-identical).
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
from starfinder.spot_finding import KNOWN_WEIGHTS, PiscisConfig, WeightsHashMismatchError
from starfinder.spot_finding import _learned
from starfinder.spot_finding._weights import weights_directory

from .learned_detectors import (SEAM_SHAPES, THREAD_VARIABLES, detect, evaluate, one_thread_environment,
    run_python, seam_scene, table_digest)
from .spot_finding_scenes import SEEDS, isolated_scene
from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset

pytestmark = [pytest.mark.extended, pytest.mark.spot_finding, pytest.mark.learned]

MODELS = ("20230905", "20251212")
SCENES = ("iso3d", "iso_z1")
# Stack mode on a 3D scene takes 12 s or more per detection on one CPU; plane mode on Z=1 does not.
SCENE_PARAMS = [pytest.param("iso3d", marks=pytest.mark.slow), "iso_z1"]
SEAM_PARAMS = [pytest.param(case, marks=pytest.mark.slow) if shape[0] > 1 else case for case, shape in SEAM_SHAPES.items()]


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


# --- S1 and S2 -------------------------------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS)
def test_s1_s2_stack_mode_on_iso3d(model, seed):
    _, truth, result = isolated("iso3d", model, seed)
    match, errors = evaluate(result.spots, truth)
    assert (result.spots.z == np.round(result.spots.z)).all()   # stack-mode z is an integer
    assert match.values["recall"] == 1.0
    assert match.values["precision"] >= 0.98
    assert errors.values["lateral_max"] <= 0.15
    assert errors.values["abs_z_max"] <= 2.0


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS)
def test_s1_s2_plane_mode_on_iso_z1(model, seed):
    _, truth, result = isolated("iso_z1", model, seed)
    match, errors = evaluate(result.spots, truth)
    assert (result.spots.z == 0).all()
    assert match.values["recall"] == 1.0
    assert match.values["precision"] >= 0.98
    assert errors.values["lateral_max"] <= 0.15


# --- S8 --------------------------------------------------------------------------------------------------

@cache
def seams(case, model, seed):
    """(truth, layout, tiled result with input_size=32, untiled native result) of a seam scene."""
    image, truth, layout = seam_scene(case, seed)
    return (truth, layout, detect(image, PiscisConfig(model, input_size=32)), detect(image, PiscisConfig(model)))


def near(spots, point, radius=3.0):
    coords = spots[["z", "y", "x"]].to_numpy()
    return coords[np.linalg.norm(coords - point, axis=1) <= radius]


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("case", SEAM_PARAMS)
def test_s8_one_candidate_per_spot_near_the_keep_boundaries(case, model, seed):
    truth, layout, tiled, untiled = seams(case, model, seed)
    assert len(truth) == 18
    for point, (kind, offset) in zip(truth, layout):
        found, reference = near(tiled.spots, point), near(untiled.spots, point)
        assert len(found) == 1, (kind, offset, found)
        assert len(reference) == 1, (kind, offset, reference)
        assert np.linalg.norm(found[0, 1:] - reference[0, 1:]) <= 0.2, (kind, offset)


@pytest.mark.validation
@pytest.mark.parametrize("case", SEAM_PARAMS)
def test_s8_the_keep_boundaries_are_recorded(case):
    _, _, tiled, untiled = seams(case, "20251212", SEEDS[0])
    mode = "plane" if case == "seam_z1" else "stack"
    assert tiled.diagnostics["geometry"] == {"mode": mode, "tile_size": [32, 32], "overlap": [3, 3],
                                             "keep_boundaries": {"y": [30.5, 59.5], "x": [30.5, 59.5]}}
    assert untiled.diagnostics["geometry"] == {"mode": mode, "tile_size": [256, 256], "overlap": [0, 0],
                                               "keep_boundaries": {"y": [], "x": []}}
    assert tiled.diagnostics["effective_settings"]["0"]["input_size"] == 32
    assert untiled.diagnostics["effective_settings"]["0"]["input_size"] == 256


# --- S13 -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def second_process_digests(torch):
    code = ("import json\n"
            "from starfinder.spot_finding import PiscisConfig\n"
            "from test.learned_detectors import detect, table_digest\n"
            "from test.spot_finding_scenes import isolated_scene\n"
            f"cases = [(s, m) for s in {SCENES!r} for m in {MODELS!r}]\n"
            "print(json.dumps({f'{s}/{m}': table_digest(detect(isolated_scene(s, 100)[0], PiscisConfig(m)).spots)"
            " for s, m in cases}))\n")
    return json.loads(run_python(code))


@pytest.mark.validation
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.slow
@pytest.mark.parametrize("scene", SCENES)
def test_s13_tables_are_identical_in_one_process_and_in_a_second(scene, model, second_process_digests):
    image, _, first = isolated(scene, model, 100)
    second = detect(image, PiscisConfig(model))
    assert len(first.spots) >= 100
    assert table_digest(first.spots) == table_digest(second.spots) == second_process_digests[f"{scene}/{model}"]


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
    entry = KNOWN_WEIGHTS[("piscis", "20251212")]
    monkeypatch.setitem(KNOWN_WEIGHTS, ("piscis", "20251212"),
                        replace(entry, files=(replace(entry.files[0], sha256="0" * 64),)))
    with pytest.raises(WeightsHashMismatchError, match="20251212.pt has SHA-256 [0-9a-f]{64}, expected 0{64}"):
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
    match, _ = evaluate(detect(image, PiscisConfig("20251212")).spots, truth)
    assert match.values["recall"] == 1.0


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
    entry = KNOWN_WEIGHTS[("piscis", model)]
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
    assert entry["artifacts"][0]["sha256"] == KNOWN_WEIGHTS[("piscis", "20251212")].sha256
