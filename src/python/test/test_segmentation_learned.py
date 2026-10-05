"""The §2.9 StarDist and Cellpose methods with their models on CPU (W-316): rows L10, L11 and L14.

Rows of the engineering validation design in docs/assignment-algorithms.md, with the
checks of docs/segmentation-contract.md ("Checks the stage wrapper applies", "Block-wise
prediction", "Tests the implementation changes"):

* L10, the ``learned`` parity test: the W-306 parity outputs, copied from
  ``parity/parity_expected.npz`` of run W-306/20261004T194036Z-74e12949 into
  ``test/data/segmentation_parity.npz`` (SHA-256 below), recomputed by the ``stardist``
  method with ``scale`` 1.0, the script's tiling, the stored thresholds and, for
  ``expand1``, planar expansion by 4 pixels; equal after a cast to ``uint16``. The
  ``rescale1`` arrays are records only (the package has no legacy round trip).
* The order of the wrapper checks for a dimensionality rejection, and the effective block
  values of block-wise prediction.
* L11: ``cellpose`` on Z=1 stains of the ``boxes`` geometry (plane z = 4, seed 102).
* L14: the cached ``2D_versatile_fluo`` files against ``KNOWN_MODELS`` and a copy with one
  byte changed.

CPU only, one thread, no download: ``2D_versatile_fluo`` and ``cpsam_v2`` come from the
weights cache (``STARFINDER_WEIGHTS_DIR``), ``3D_spleen`` from the folder in
``STARFINDER_STARDIST_3D_SPLEEN``. A test skips with its reason when its extra or its model
is absent.
"""
import hashlib
import os
import pickle
import shutil
import socket
import subprocess
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation import (KNOWN_MODELS, SEGMENTATION_METHODS, CellposeConfig, ExpandLabelsConfig,
    MissingModelError, ModelHashMismatchError, ReferenceGrid, SegmentationInput, StarDistConfig, expand_labels,
    resolve_model, segment)
from starfinder.spot_finding._weights import weights_directory

from .learned_detectors import one_thread_environment
from .segmentation_fixtures import BOXES_METADATA, boxes

pytestmark = [pytest.mark.segmentation, pytest.mark.learned]

PARITY = Path(__file__).parent / "data" / "segmentation_parity.npz"
PARITY_SHA256 = "4a5df9973e0f487920aeed2dd9523bed1a929d020695f95c6c212f61be443e0f"
NAMESPACE = '["parity",null,null,null,"nucleus"]'
SPLEEN_VARIABLE = "STARFINDER_STARDIST_3D_SPLEEN"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def parity():
    assert sha256(PARITY.read_bytes()) == PARITY_SHA256
    with np.load(PARITY) as data:
        return {name: data[name] for name in data.files}


@pytest.fixture(scope="module")
def tensorflow():
    pytest.importorskip("stardist", reason="needs the stardist extra")
    return pytest.importorskip("tensorflow", reason="needs the stardist extra")


def cached(method, model):
    folder = weights_directory() / method / model
    if not all((folder / item.path).is_file() for item in KNOWN_MODELS[(method, model)].files):
        pytest.skip(f"{method} model {model} is not in the weights cache {weights_directory()}; fetch it with "
                    f"'starfinder weights fetch {method} {model}'")
    return folder


@pytest.fixture(scope="module")
def versatile():
    cached("stardist", "2D_versatile_fluo")
    return StarDistConfig(model="2D_versatile_fluo", scale=1.0)


@pytest.fixture(scope="module")
def spleen():
    path = os.environ.get(SPLEEN_VARIABLE)
    if not path or not Path(path, "weights_best.h5").is_file():
        pytest.skip(f"the user-trained 3D_spleen model is given by {SPLEEN_VARIABLE}, which is unset or names no "
                    "model folder")
    return StarDistConfig(model_path=path, scale=1.0)


def zyx(image):
    return image if image.ndim == 3 else image[np.newaxis]


def parity_input(image, name):
    image = zyx(image)
    grid = ReferenceGrid(image.shape, ImageMetadata(f"parity_{name}"), "declared")
    return SegmentationInput(image[..., np.newaxis], grid, ("nuclear",))


def check_parity(parity, name, config):
    """The two rescale0 arrays of one parity input, recomputed and compared after a cast to uint16."""
    image = parity[f"input__{name}"]
    result = segment(parity_input(image, name), config=config, target="nucleus", label_namespace=NAMESPACE)
    assert result.labels.dtype == np.uint32 and result.labels.shape == zyx(image).shape
    assert result.grid.shape_zyx == zyx(image).shape
    effective = result.record["methods"][0]["effective"]
    assert effective["threshold_source"] == "stored" and effective["prediction"] == "whole_image"
    expanded, _ = expand_labels(result.labels, result.grid.metadata,
                                config=ExpandLabelsConfig(distance=4, unit="pixel", mode="planar"))
    for expand, labels in ((0, result.labels), (1, expanded)):
        expected = parity[f"script__{name}__rescale0_expand{expand}"]
        assert np.array_equal(labels.reshape(expected.shape).astype(np.uint16), expected), (name, expand)
    return result


# --- L10: stardist parity ------------------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.golden
def test_l10_2d_versatile_fluo_reproduces_the_parity_outputs(parity, tensorflow, versatile):
    for name, n_labels in (("P2_tissue_pi_2d", 23), ("P3_discs_61x67_2d", 5)):
        result = check_parity(parity, name, versatile)
        assert result.n_labels == n_labels and result.geometry == "plane"
        entry = result.record["methods"][0]
        assert entry["effective"]["n_tiles"] == [2, 2] and entry["effective"]["scale_zyx"] == [1.0, 1.0, 1.0]
        assert (entry["effective"]["prob_thresh"], entry["effective"]["nms_thresh"]) == (0.479071463157368, 0.3)
        assert [a["sha256"] for a in entry["artifacts"]] == [
            f.sha256 for f in KNOWN_MODELS[("stardist", "2D_versatile_fluo")].files]
        assert entry["execution"]["device"] == "cpu" and entry["execution"]["framework"]["name"] == "tensorflow"


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.golden
def test_l10_3d_spleen_reproduces_the_parity_outputs(parity, tensorflow, spleen):
    result = check_parity(parity, "P1_ln_dapi_round4_3d", spleen)
    assert result.n_labels == 13 and result.geometry == "volume"
    effective = result.record["methods"][0]["effective"]
    assert effective["n_tiles"] == [1, 4, 4]
    assert (effective["prob_thresh"], effective["nms_thresh"]) == (0.6428984851984139, 0.5)
    assert [a["source"] for a in result.record["methods"][0]["artifacts"]] == ["path"] * 3


@pytest.mark.contract
@pytest.mark.validation
def test_l10_a_model_of_the_wrong_dimensionality_is_rejected_at_check_7(parity, tensorflow, versatile, spleen,
                                                                       monkeypatch):
    spec = SEGMENTATION_METHODS[StarDistConfig]
    calls = []
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig,
                        replace(spec, run=lambda *args: calls.append(args) or spec.run(*args)))
    volume = parity["input__P1_ln_dapi_round4_3d"]
    cases = ((spleen, volume[:1], "a 3D model cannot segment an input with Z=1"),
             (versatile, volume, "a 2D model cannot segment an input with Z=16"))
    for config, image, message in cases:
        with pytest.raises(IncompatibleGeometryError, match=message):
            segment(parity_input(image, "z"), config=config, target="nucleus", label_namespace=NAMESPACE)
    # Rejected after the dependency import (check 5) and before the method's run (check 9): no model was built.
    assert calls == []
    assert "tensorflow" in sys.modules and "stardist" in sys.modules


@pytest.mark.validation
def test_block_wise_prediction_records_the_requested_and_effective_block_values(parity, tensorflow):
    cached("stardist", "2D_versatile_fluo")
    requested = dict(block_size=(255, 257), min_overlap=(33, 35), context=(15, 17))
    config = StarDistConfig(model="2D_versatile_fluo", scale=1.0, **requested)
    result = segment(parity_input(parity["input__P2_tissue_pi_2d"], "P2"), config=config, target="nucleus",
                     label_namespace=NAMESPACE)
    effective = result.record["methods"][0]["effective"]
    assert effective["prediction"] == "block_wise" and effective["blocks"]["grid"] == [2, 2]
    assert effective["blocks"]["requested"] == {name: list(value) for name, value in requested.items()}
    assert effective["blocks"]["effective"] == {"block_size": [256, 258], "min_overlap": [34, 36],
                                                "context": [16, 18]}
    assert all(v % 2 == 0 for values in effective["blocks"]["effective"].values() for v in values)
    assert result.record["methods"][0]["config"]["block_size"] == [255, 257]
    assert result.labels.shape == (1, 512, 512) and result.labels.dtype == np.uint32


# --- L11: cellpose -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cellpose_model():
    pytest.importorskip("cellpose", reason="needs the cellpose extra")
    pytest.importorskip("torch", reason="needs the cellpose extra")
    cached("cellpose", "cpsam_v2")
    return CellposeConfig(model="cpsam_v2", diameter=10.0)


def l11_input():
    """Z=1 stains of the boxes plane z = 4: 200 in cells (cytoplasm) or nuclei (nuclear), 20 elsewhere, noise σ 5."""
    cells, nuclei = boxes()
    rng = np.random.default_rng(102)
    stains = [np.where(mask[4] > 0, 200.0, 20.0) + rng.normal(0, 5, mask[4].shape) for mask in (cells, nuclei)]
    image = np.clip(np.rint(np.stack(stains, axis=-1)), 0, 255).astype(np.uint8)[np.newaxis]
    grid = ReferenceGrid((1, *cells.shape[1:]), BOXES_METADATA.projected(method="max"), "declared")
    return SegmentationInput(image, grid, ("cytoplasm", "nuclear"))


# Two Cellpose calls in one process of their own: cpsam_v2 holds about 2.5 GiB while it loads, and the pytest process
# may already hold TensorFlow, so the calls run beside it instead (one learned backend per process).
L11_CALLS = """
import pickle, sys
from cellpose import models
from starfinder.segmentation import segment
with open(sys.argv[1], "rb") as handle:
    segmentation_input, config, namespace = pickle.load(handle)
defaults = dict(models.normalize_default)
results = [segment(segmentation_input, config=config, target="cell", label_namespace=namespace) for _ in range(2)]
with open(sys.argv[2], "wb") as handle:
    pickle.dump({"results": [(r.labels, r.geometry, dict(r.record)) for r in results],
                 "defaults_kept": models.normalize_default == defaults}, handle)
"""


@pytest.mark.slow
@pytest.mark.validation
def test_l11_cellpose_on_a_plane_is_repeatable_and_uint32(cellpose_model, tmp_path):
    source, target = tmp_path / "l11_input.pkl", tmp_path / "l11_results.pkl"
    with open(source, "wb") as handle:
        pickle.dump((l11_input(), cellpose_model, NAMESPACE), handle)
    result = subprocess.run([sys.executable, "-c", L11_CALLS, str(source), str(target)], capture_output=True,
                            text=True, env=one_thread_environment(), cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr[-4000:]
    with open(target, "rb") as handle:
        out = pickle.load(handle)
    (first, geometry, record), (second, _, _) = out["results"]
    assert first.shape == (1, 32, 32) and first.dtype == np.uint32 and geometry == "plane"
    assert np.array_equal(first, second)
    assert out["defaults_kept"]                         # the normalization mapping is not mutated
    entry = record["methods"][0]
    assert entry["effective"]["channels"] == ["cytoplasm", "nuclear"]
    assert (entry["effective"]["tile_overlap"], entry["effective"]["bfloat16"]) == (0.1, True)
    assert entry["execution"]["device"] == "cpu" and entry["execution"]["framework"]["name"] == "torch"
    assert [a["sha256"] for a in entry["artifacts"]] == [KNOWN_MODELS[("cellpose", "cpsam_v2")].sha256]


@pytest.mark.contract
@pytest.mark.validation
def test_l11_a_missing_cellpose_model_raises_before_any_library_call(cellpose_model, tmp_path, monkeypatch):
    from cellpose import models
    built = []
    monkeypatch.setattr(models, "CellposeModel", lambda *a, **k: built.append((a, k)))
    with pytest.raises(MissingModelError, match="does not exist; segmentation never downloads"):
        segment(l11_input(), config=CellposeConfig(model_path=str(tmp_path / "cpsam_v2"), diameter=10.0),
                target="cell", label_namespace=NAMESPACE)
    assert built == []
    with pytest.raises(TypeError):
        CellposeConfig(model="cpsam_v2")               # diameter is required


# --- L14: model resolution with the cached files -------------------------------------------------------

@pytest.mark.validation
def test_l14_the_cached_files_equal_the_table_and_a_changed_copy_raises(tmp_path, monkeypatch):
    folder = cached("stardist", "2D_versatile_fluo")
    connections, opened = [], []
    monkeypatch.setattr(socket.socket, "connect", lambda *a: connections.append(a))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: opened.append(a))
    entry = KNOWN_MODELS[("stardist", "2D_versatile_fluo")]
    path, artifacts = resolve_model("stardist", model="2D_versatile_fluo")
    assert path == folder and [(a["name"], a["sha256"]) for a in artifacts] == [
        (f"stardist/2D_versatile_fluo/{f.path}", f.sha256) for f in entry.files]
    assert [sha256((folder / f.path).read_bytes()) for f in entry.files] == [f.sha256 for f in entry.files]

    copy = tmp_path / "weights" / "stardist" / "2D_versatile_fluo"
    shutil.copytree(folder, copy)
    data = bytearray((copy / "weights_best.h5").read_bytes())
    data[len(data) // 2] ^= 0x01
    (copy / "weights_best.h5").write_bytes(bytes(data))
    changed = sha256(bytes(data))
    with pytest.raises(ModelHashMismatchError) as raised:
        resolve_model("stardist", model="2D_versatile_fluo", directory=tmp_path / "weights")
    assert changed in str(raised.value) and entry.files[2].sha256 in str(raised.value)
    # The same through segment, before the method runs: nothing is fetched and no model is built.
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(tmp_path / "weights"))
    spec = SEGMENTATION_METHODS[StarDistConfig]
    calls = []
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig,
                        replace(spec, requires=(), run=lambda *a: calls.append(a)))
    image = np.zeros((1, 8, 8, 1), np.uint8)
    with pytest.raises(ModelHashMismatchError, match=changed):
        segment(SegmentationInput(image, ReferenceGrid((1, 8, 8), ImageMetadata("l14"), "declared"), ("nuclear",)),
                config=StarDistConfig(model="2D_versatile_fluo", scale=1.0), target="nucleus",
                label_namespace=NAMESPACE)
    (copy / "weights_best.h5").unlink()
    with pytest.raises(MissingModelError, match="starfinder weights fetch stardist 2D_versatile_fluo"):
        resolve_model("stardist", model="2D_versatile_fluo")
    assert calls == [] and connections == [] and opened == []
