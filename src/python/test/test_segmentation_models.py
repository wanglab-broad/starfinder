"""The §2.9 StarDist and Cellpose configs, the known-models table, model resolution and the fetch command (W-316).

Default tier: nothing here imports StarDist, TensorFlow, Cellpose or torch, and no model
file of the weights cache is read. The configs and the block fields follow
docs/segmentation-contract.md ("Registered methods", "Block-wise prediction"); the table,
resolution and the fetch command follow its "Models" section; the behavior without the
extras follows checks 5 and "Method registry". Model folders are small files written in
the test, and the fetch downloads a local zip through a ``file://`` URL, so no network is
used. The learned checks L10, L11 and L14 are in ``test_segmentation_learned.py``.
"""
import hashlib
import json
import socket
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from starfinder.__main__ import main
from starfinder._registry import names
from starfinder.image import ImageMetadata
from starfinder.segmentation import (KNOWN_MODELS, SEGMENTATION_METHODS, CellposeConfig, KnownModel, MissingModelError,
    ModelFile, ModelHashMismatchError, ReferenceGrid, SegmentationBackendUnavailableError, SegmentationInput,
    StarDistConfig, resolve_model, segment)
from starfinder.segmentation import _models

pytestmark = [pytest.mark.segmentation, pytest.mark.contract]

NAMESPACE = '["data","sample","FOV_001",null,"nucleus"]'
BACKENDS = ("stardist", "csbdeep", "tensorflow", "cellpose", "torch")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty weights cache; any network connection fails the test."""
    root = tmp_path / "weights"
    root.mkdir()
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(root))

    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    return root


# --- The known-models table ---------------------------------------------------------------------------

def test_the_known_models_table_holds_the_contract_rows():
    assert set(KNOWN_MODELS) == {("stardist", "2D_versatile_fluo"), ("cellpose", "cpsam_v2")}
    fluo = KNOWN_MODELS[("stardist", "2D_versatile_fluo")]
    assert fluo.files == (
        ModelFile("config.json", "836da16282c3e0db1ba2e58f377e977419887ace8c737ebde16a094d58d50f74", 1021),
        ModelFile("thresholds.json", "5cd6aac6e923f8659b63a9e297485920d45a59dad35371ddb8b6ae4be398d805", 39),
        ModelFile("weights_best.h5", "42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349", 5771480))
    assert (fluo.url, fluo.sha256, fluo.archive, fluo.grid) == (
        "https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip",
        "4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd", True, (2, 2))
    assert fluo.stored_thresholds == {"prob": 0.479071463157368, "nms": 0.3}
    cpsam = KNOWN_MODELS[("cellpose", "cpsam_v2")]
    assert cpsam.files == (ModelFile("cpsam_v2", "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
                                     1233586851),)
    assert (cpsam.url, cpsam.sha256, cpsam.bytes, cpsam.archive, cpsam.stored_thresholds) == (
        "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2",
        "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667", 1233586851, False, None)


# --- The configs ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("blocks", [
    dict(block_size=64), dict(min_overlap=8), dict(context=8),
    dict(block_size=64, min_overlap=8), dict(block_size=64, context=8), dict(min_overlap=8, context=8)])
def test_block_fields_are_given_together(blocks, monkeypatch):
    # The config raises before any backend could load a model: the backends cannot even be imported here.
    for module in BACKENDS:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ValueError, match="given together"):
        StarDistConfig(model="2D_versatile_fluo", scale=1.0, **blocks)


@pytest.mark.parametrize("blocks", [
    dict(block_size=(64, 64), min_overlap=(8, 32), context=(8, 16)),       # X: 32 + 32 = 64
    dict(block_size=(20, 64, 64), min_overlap=(4, 8, 8), context=(8, 4, 4)),  # Z: 4 + 16 > 20
    dict(block_size=48, min_overlap=16, context=16)])
def test_block_fields_need_min_overlap_plus_twice_the_context_below_the_block_size(blocks, monkeypatch):
    for module in BACKENDS:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ValueError, match=r"min_overlap \+ 2 × context < block_size"):
        StarDistConfig(model="2D_versatile_fluo", scale=1.0, **blocks)


@pytest.mark.parametrize("scale", [0.5, (1.0, 0.5, 0.5), (0.5, 1.0, 1.0)])
def test_block_fields_with_a_scale_other_than_one_raise(scale, monkeypatch):
    for module in BACKENDS:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ValueError, match="scale other than 1"):
        StarDistConfig(model="2D_versatile_fluo", scale=scale, block_size=64, min_overlap=8, context=8)


def test_stardist_config_fields():
    config = StarDistConfig(model="2D_versatile_fluo", scale=1.0)
    assert (config.method, config.model_path, config.prob_thresh, config.nms_thresh, config.normalize_percentiles,
            config.n_tiles, config.block_size, config.min_overlap, config.context) == (
        "stardist", None, None, None, (1.0, 99.8), None, None, None, None)
    blocks = StarDistConfig(model_path="/models/3D_spleen", scale=(1.0, 1.0, 1.0), block_size=(50, 256, 256),
                            min_overlap=(0, 64, 64), context=(0, 32, 32))
    assert blocks.block_size == (50, 256, 256)
    with pytest.raises(TypeError):
        StarDistConfig(model="2D_versatile_fluo")      # scale has no default
    for change in (dict(), dict(model="2D_versatile_fluo", model_path="/m"), dict(model="2D_versatile_fluo_x"),
                   dict(model="2D_brain_overlay_05"), dict(model="2D_versatile_fluo", scale=0),
                   dict(model="2D_versatile_fluo", scale=(1.0, 0.5)), dict(model="2D_versatile_fluo", prob_thresh=1.5),
                   dict(model="2D_versatile_fluo", normalize_percentiles=(99.8, 1.0)),
                   dict(model="2D_versatile_fluo", n_tiles=(2,)), dict(model="2D_versatile_fluo", n_tiles=(0, 2)),
                   dict(model="2D_versatile_fluo", model_sha256={"config.json": "0" * 64}),
                   dict(model="2D_versatile_fluo", block_size=64, min_overlap=-1, context=8),
                   dict(model="2D_versatile_fluo", block_size=(64, 64), min_overlap=(8, 8, 8), context=8)):
        change.setdefault("scale", 1.0)
        with pytest.raises(ValueError):
            StarDistConfig(**change)


def test_cellpose_config_fields():
    config = CellposeConfig(model="cpsam_v2", diameter=30.0)
    assert (config.method, config.do_3d, config.anisotropy, config.flow_threshold, config.cellprob_threshold,
            config.tile_overlap, config.bfloat16, config.min_size, config.normalize_percentiles) == (
        "cellpose", False, None, 0.4, 0.0, 0.1, True, 15, (1.0, 99.0))
    assert CellposeConfig(model="cpsam_v2", diameter=None).diameter is None
    assert CellposeConfig(model="cpsam_v2", diameter=240.0, do_3d=True, anisotropy=2.0).do_3d
    with pytest.raises(TypeError):
        CellposeConfig(model="cpsam_v2")                # diameter is required
    for change in (dict(diameter=None, do_3d=True, anisotropy=2.0), dict(diameter=30.0, do_3d=True),
                   dict(diameter=30.0, anisotropy=2.0), dict(diameter=0.0), dict(diameter=30.0, tile_overlap=0.6),
                   dict(diameter=30.0, min_size=-2), dict(diameter=30.0, model="cyto3"),
                   dict(diameter=30.0, model_path="/m"), dict(diameter=30.0, do_3d="yes")):
        change.setdefault("model", "cpsam_v2")          # with model_path: both are given
        with pytest.raises(ValueError):
            CellposeConfig(**change)


# --- The registry and the behavior without the extras ---------------------------------------------------

def test_the_registry_lists_the_three_methods_with_their_contract_fields():
    assert set(names(SEGMENTATION_METHODS)) == {"stardist", "cellpose", "seeded_watershed"}
    stardist, cellpose = SEGMENTATION_METHODS[StarDistConfig], SEGMENTATION_METHODS[CellposeConfig]
    assert (stardist.targets, stardist.roles, stardist.required_roles, stardist.seeds, stardist.dimensions,
            stardist.models, stardist.devices, stardist.min_shape_zyx) == (
        frozenset({"nucleus", "cell"}), frozenset({"nuclear", "composite"}), (frozenset({"nuclear", "composite"}),),
        "none", frozenset({2, 3}), True, frozenset({"cpu", "cuda"}), (1, 1, 1))
    assert [(d.module, d.extra) for d in stardist.requires] == [
        ("stardist", "stardist"), ("csbdeep", "stardist"), ("tensorflow", "stardist")]
    assert (cellpose.targets, cellpose.roles, cellpose.required_roles, cellpose.seeds, cellpose.dimensions,
            cellpose.models, cellpose.devices) == (
        frozenset({"nucleus", "cell"}), frozenset({"cytoplasm", "nuclear"}), (frozenset({"cytoplasm", "nuclear"}),),
        "none", frozenset({2, 3}), True, frozenset({"cpu", "cuda"}))
    assert [(d.module, d.extra) for d in cellpose.requires] == [("cellpose", "cellpose"), ("torch", "cellpose")]


def plane_input(roles=("nuclear",)):
    image = np.zeros((1, 16, 16, len(roles)), np.uint8)
    return SegmentationInput(image, ReferenceGrid((1, 16, 16), ImageMetadata("plane"), "declared"), roles)


@pytest.mark.parametrize("config, extra", [
    (StarDistConfig(model="2D_versatile_fluo", scale=1.0), "stardist"),
    (CellposeConfig(model="cpsam_v2", diameter=30.0), "cellpose")])
def test_without_the_extras_a_learned_method_names_its_extra(config, extra, monkeypatch):
    # The locked environment without the extras: the backend imports fail (guidance: monkeypatch sys.modules).
    for module in BACKENDS:
        monkeypatch.setitem(sys.modules, module, None)
    assert set(names(SEGMENTATION_METHODS)) == {"stardist", "cellpose", "seeded_watershed"}
    with pytest.raises(SegmentationBackendUnavailableError, match=rf"install the '{extra}' extra"):
        segment(plane_input(), config=config, target="nucleus", label_namespace=NAMESPACE)


def test_the_package_imports_no_backend():
    code = ("import sys, starfinder.segmentation, starfinder.__main__; "
            "print(sorted(m for m in ('stardist', 'csbdeep', 'tensorflow', 'cellpose', 'torch') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# --- Model resolution -----------------------------------------------------------------------------------

FILES = {"config.json": b'{"n_dim": 2, "grid": [2, 2]}', "thresholds.json": b'{"prob": 0.5, "nms": 0.4}',
         "weights_best.h5": b"fixture weights"}


def fixture_entry(url="file:///absent.zip", data=b""):
    return KnownModel("stardist", "fixture_model", url, sha256(data), len(data), True,
                      tuple(ModelFile(name, sha256(content), len(content)) for name, content in FILES.items()),
                      "2D YX", "1", (2, 2), {"prob": 0.5, "nms": 0.4}, "none", "test fixture")


def write_folder(folder, files=FILES):
    folder.mkdir(parents=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    return folder


def test_a_known_model_resolves_in_the_cache_with_its_hashes(cache, monkeypatch):
    monkeypatch.setitem(KNOWN_MODELS, ("stardist", "fixture_model"), fixture_entry())
    folder = write_folder(cache / "stardist" / "fixture_model")
    path, artifacts = resolve_model("stardist", model="fixture_model")
    assert path == folder.resolve()
    assert [(a["name"], a["sha256"], a["source"]) for a in artifacts] == [
        (f"stardist/fixture_model/{name}", sha256(content), "cache") for name, content in FILES.items()]


def test_a_missing_or_changed_known_model_file_raises(cache, monkeypatch):
    monkeypatch.setitem(KNOWN_MODELS, ("stardist", "fixture_model"), fixture_entry())
    with pytest.raises(MissingModelError, match="fetch it with 'starfinder weights fetch stardist fixture_model'"):
        resolve_model("stardist", model="fixture_model")
    folder = write_folder(cache / "stardist" / "fixture_model")
    (folder / "weights_best.h5").write_bytes(b"fixture weightz")
    with pytest.raises(ModelHashMismatchError) as raised:
        resolve_model("stardist", model="fixture_model")
    assert sha256(b"fixture weightz") in str(raised.value) and sha256(FILES["weights_best.h5"]) in str(raised.value)
    with pytest.raises(ValueError, match="unknown stardist model '3D_spleen'"):
        resolve_model("stardist", model="3D_spleen")
    with pytest.raises(ValueError, match="exactly one of model and model_path"):
        resolve_model("stardist")


def test_a_user_model_by_path_is_hashed_and_checked(tmp_path):
    folder = write_folder(tmp_path / "user_model", dict(FILES, **{"weights_last.h5": b"not loaded"}))
    path, artifacts = resolve_model("stardist", model_path=folder)
    assert path == folder.resolve()
    # Only the three files StarDist loads are hashed and recorded.
    assert [(a["name"], a["sha256"], a["source"]) for a in artifacts] == [
        (f"stardist/user_model/{name}", sha256(content), "path") for name, content in FILES.items()]
    expected = {"weights_best.h5": sha256(FILES["weights_best.h5"])}
    assert resolve_model("stardist", model_path=folder, model_sha256=expected)[0] == folder.resolve()
    with pytest.raises(ModelHashMismatchError, match=f"differs from the expected {'0' * 64}"):
        resolve_model("stardist", model_path=folder, model_sha256={"weights_best.h5": "0" * 64})
    (folder / "thresholds.json").unlink()
    with pytest.raises(MissingModelError, match="thresholds.json"):
        resolve_model("stardist", model_path=folder)
    with pytest.raises(MissingModelError, match="does not exist; segmentation never downloads"):
        resolve_model("cellpose", model_path=tmp_path / "absent")
    with pytest.raises(MissingModelError, match="is not a file"):
        resolve_model("cellpose", model_path=folder)
    model_file = tmp_path / "cellpose_model"
    model_file.write_bytes(b"cellpose fixture")
    path, artifacts = resolve_model("cellpose", model_path=model_file)
    assert path == model_file.resolve() and artifacts == [
        {"name": "cellpose/cellpose_model", "path": str(model_file.resolve()), "sha256": sha256(b"cellpose fixture"),
         "source": "path"}]


def test_segment_resolves_the_model_before_the_backend_runs(cache, tmp_path, monkeypatch):
    # Check 6 raises for a missing user model; the method's run is never reached.
    spec = SEGMENTATION_METHODS[StarDistConfig]
    calls = []
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig,
                        replace(spec, requires=(), run=lambda *a: calls.append(a)))
    with pytest.raises(MissingModelError, match="does not exist"):
        segment(plane_input(), config=StarDistConfig(model_path=str(tmp_path / "absent"), scale=1.0),
                target="nucleus", label_namespace=NAMESPACE)
    assert calls == []


# --- The fetch command ----------------------------------------------------------------------------------

@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A StarDist-like model zip on disk and its table entry; _download is counted."""
    path = tmp_path / "source" / "fixture_model.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as handle:
        for name, content in FILES.items():
            handle.writestr(name, content)
    entry = fixture_entry(path.as_uri(), path.read_bytes())
    monkeypatch.setitem(KNOWN_MODELS, ("stardist", "fixture_model"), entry)
    from starfinder.spot_finding import _fetch
    urls, download = [], _fetch._download

    def counted(url, target, *args):
        urls.append(url)
        return download(url, target, *args)
    monkeypatch.setattr(_fetch, "_download", counted)
    return entry, urls


def test_weights_fetch_installs_a_segmentation_model(cache, archive, capsys):
    entry, urls = archive
    assert main(["weights", "list"]) == 0
    assert "stardist\tfixture_model\t" in capsys.readouterr().out
    assert main(["weights", "fetch", "stardist", "fixture_model"]) == 0
    folder = cache / "stardist" / "fixture_model"
    assert capsys.readouterr().out.strip() == str(folder) and urls == [entry.url]
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.name != _models.RECORD_NAME} == FILES
    record = json.loads((folder / _models.RECORD_NAME).read_text())
    assert record["sha256"] == entry.sha256 and record["files"] == {f.path: f.sha256 for f in entry.files}
    assert main(["weights", "verify", "stardist", "fixture_model"]) == 0
    assert "stardist\tfixture_model\tverified" in capsys.readouterr().out
    # A verified copy is kept: no second download.
    assert main(["weights", "fetch", "stardist", "fixture_model"]) == 0 and urls == [entry.url]
    capsys.readouterr()
    (folder / "thresholds.json").write_bytes(b"changed")
    assert main(["weights", "verify"]) == 1
    assert "stardist\tfixture_model\tfailed" in capsys.readouterr().out
    with pytest.raises(SystemExit) as raised:      # a changed file is never overwritten
        main(["weights", "fetch", "stardist", "fixture_model"])
    assert raised.value.code == 2 and (folder / "thresholds.json").read_bytes() == b"changed"


def test_weights_fetch_restores_a_missing_file_and_refuses_a_bad_download(cache, archive, monkeypatch):
    entry, urls = archive
    folder = _models._fetch_model("stardist", "fixture_model")
    (folder / "weights_best.h5").unlink()
    assert _models._fetch_model("stardist", "fixture_model") == folder and len(urls) == 2
    assert (folder / "weights_best.h5").read_bytes() == FILES["weights_best.h5"]
    monkeypatch.setitem(KNOWN_MODELS, ("stardist", "fixture_model"), replace(entry, sha256="0" * 64))
    (folder / "weights_best.h5").unlink()
    with pytest.raises(ModelHashMismatchError, match=f"expected {entry.bytes} bytes with SHA-256 {'0' * 64}"):
        _models._fetch_model("stardist", "fixture_model")
    assert not (folder / "weights_best.h5").exists()
    assert [p.name for p in (cache / "stardist").iterdir()] == ["fixture_model"]   # no staging folder is left


def test_no_segmentation_entry_downloads(cache, monkeypatch):
    # The resolution of a known model that is absent names the fetch command and opens no URL.
    opened = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: opened.append(a))
    with pytest.raises(MissingModelError, match="starfinder weights fetch stardist 2D_versatile_fluo"):
        resolve_model("stardist", model="2D_versatile_fluo")
    spec = SEGMENTATION_METHODS[CellposeConfig]
    monkeypatch.setitem(SEGMENTATION_METHODS, CellposeConfig, replace(spec, requires=()))
    with pytest.raises(MissingModelError, match="starfinder weights fetch cellpose cpsam_v2"):
        segment(plane_input(("cytoplasm",)), config=CellposeConfig(model="cpsam_v2", diameter=30.0),
                target="cell", label_namespace=NAMESPACE)
    assert opened == []
