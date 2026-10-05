"""The §2.9 engineering validation of segmentation: the L checks the task groups did not add (W-318).

Rows of the engineering validation design in docs/assignment-algorithms.md ("Checks"). Rows L1
to L11, L13 and L14 are in the modules of their task groups; the section "Implemented checks"
of that page maps every row to its tests, and ``test_every_l_row_names_existing_tests`` keeps
that map true. This module adds:

* L4's identity, empty-image and missing-spacing clauses on the row's own fixtures, the
  ``seg_golden`` stand-in labels and the single voxel in 9×21×21 with spacing (0.3, 0.1, 0.1)
  (the earlier tests check them on ``boxes``);
* L7 on ``seg_golden``: ``rescale_input`` of the W-307 golden DAPI image with a spacing divides
  the spacing by the factors, records the rescale in ``frame_id`` and gives W-307's pinned
  shrunk image (the ramp of ``test_segmentation_inputs.py`` checks the voxel centres);
* L12, CPU against GPU on the parity inputs P1 and P2 (``learned``, ``slow``). It runs only in a
  batch whose guidance grants the GPU: ``STARFINDER_L12_GPU_PYTHON`` names the Python of the GPU
  environment, and without it the test skips with that reason. Each device runs in a process of
  its own (one learned backend per process); the CPU process is the reference. The bounds are
  the W-306 tolerances, unchanged: label count within max(1, 1 %), at least 99 % of the CPU
  labels matched at IoU ≥ 0.5, median matched IoU at least 0.99.

Fixtures are hand-built or pinned (``seg_golden``, seed 20261005; the parity file); no other
random number is used.
"""
import ast
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from starfinder.image import ImageMetadata
from starfinder.segmentation import KNOWN_MODELS, ExpandLabelsConfig, expand_labels, rescale_input
from starfinder.spot_finding._weights import weights_directory

from .learned_detectors import one_thread_environment
from .test_segmentation_golden import SHAPE_ZYX, SHRUNK_IMAGE_DIGEST, digest, fixture, stand_in_model

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

TEST_ROOT = Path(__file__).resolve().parent
DESIGN = TEST_ROOT.parents[2] / "docs" / "assignment-algorithms.md"
PARITY = TEST_ROOT / "data" / "segmentation_parity.npz"
PARITY_SHA256 = "4a5df9973e0f487920aeed2dd9523bed1a929d020695f95c6c212f61be443e0f"
GPU_PYTHON = "STARFINDER_L12_GPU_PYTHON"
SPLEEN_VARIABLE = "STARFINDER_STARDIST_3D_SPLEEN"


# --- The map of the section "Implemented checks" ---------------------------------------------------

def implemented_checks():
    """{row: the backticked names of its "Module and tests" cell} from the design page."""
    section = DESIGN.read_text().split("### Implemented checks", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"\| ([LA]\d+) \| ([^|]+) \|", line)
        if match:
            rows[match.group(1)] = re.findall(r"`([^`]+)`", match.group(2))
    return rows


def defined_tests(module):
    tree = ast.parse((TEST_ROOT / module).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def unmatched_names(names):
    """The test names of one row that no function of the module named before them matches."""
    module, missing = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
            continue
        if not name.startswith("test_"):
            continue
        pattern = name.split("[", 1)[0]
        defined = defined_tests(module) if module else set()
        found = any(f.startswith(pattern[:-1]) for f in defined) if pattern.endswith("*") else pattern in defined
        if not found:
            missing.append((module, name))
    return missing


def check_rows(prefix, count):
    rows = implemented_checks()
    expected = [f"{prefix}{i}" for i in range(1, count + 1)]
    assert sorted(r for r in rows if r.startswith(prefix)) == sorted(expected)
    for row in expected:
        assert any(n.startswith("test_") for n in rows[row]), row
        assert unmatched_names(rows[row]) == [], row


def test_every_l_row_names_existing_tests():
    check_rows("L", 14)


# --- L4: expand_labels on the row's fixtures --------------------------------------------------------

def l4_fixtures():
    """(name, labels, metadata): the seg_golden stand-in labels (uncalibrated, as W-307) and the single voxel."""
    seed = np.zeros((9, 21, 21), np.uint32)
    seed[4, 10, 10] = 1
    return (("seg_golden", stand_in_model(fixture()["dapi"]), ImageMetadata("seg_golden")),
            ("single_voxel", seed, ImageMetadata("single", spacing_zyx=(0.3, 0.1, 0.1))))


@pytest.mark.parametrize("mode", ["planar", "volumetric"])
def test_l4_distance_zero_is_the_identity_and_an_empty_image_stays_empty(mode):
    for name, labels, metadata in l4_fixtures():
        units = ("pixel", "um") if metadata.spacing_zyx else ("pixel",)
        for unit in units:
            unchanged, record = expand_labels(labels, metadata, config=ExpandLabelsConfig(0, unit, mode))
            assert unchanged.dtype == np.uint32 and np.array_equal(unchanged, labels), (name, unit)
            assert record["voxels_added"] == 0
            empty, record = expand_labels(np.zeros_like(labels), metadata, config=ExpandLabelsConfig(4, unit, mode))
            assert empty.dtype == np.uint32 and empty.shape == labels.shape and not empty.any(), (name, unit)
            assert record["voxels_added"] == 0


@pytest.mark.parametrize("mode", ["planar", "volumetric"])
def test_l4_um_without_spacing_raises(mode):
    for name, labels, metadata in l4_fixtures():
        for missing in (None, ImageMetadata(name)):
            with pytest.raises(ValueError, match="spacing_zyx"):
                expand_labels(labels, missing, config=ExpandLabelsConfig(0.5, "um", mode))


# --- L7: rescale_input on seg_golden ---------------------------------------------------------------

def test_l7_rescaling_seg_golden_divides_the_spacing_and_records_the_rescale():
    dapi = fixture()["dapi"]
    metadata = ImageMetadata("seg_golden", spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
    output, rescaled, record = rescale_input(dapi, metadata, scale_zyx=(1, 0.5, 0.5))
    assert output.shape == (SHAPE_ZYX[0], SHAPE_ZYX[1] // 2, SHAPE_ZYX[2] // 2)
    assert digest(output) == SHRUNK_IMAGE_DIGEST
    assert rescaled.spacing_zyx == (0.35 / 1, 0.1 / 0.5, 0.1 / 0.5) == (0.35, 0.2, 0.2)
    assert rescaled.frame_id == "seg_golden/rescale:1.0,0.5,0.5" == record["frame_id"]
    assert rescaled.spatial_unit == "micrometer" and record["config"] == {"scale_zyx": [1.0, 0.5, 0.5]}
    assert record["inputs"] == [digest(dapi)] and record["output"] == digest(output)


# --- L12: CPU against GPU ----------------------------------------------------------------------------

def l12_agreement(reference, other):
    """(labels in reference, labels in other, matched fraction of the reference labels, median matched IoU).

    Pairs at IoU ≥ 0.5 are matched one to one, the larger IoU first (two pairs can share a label only
    at exactly 0.5).
    """
    a, b = np.asarray(reference).ravel(), np.asarray(other).ravel()
    ids_a, size_a = np.unique(a[a > 0], return_counts=True)
    ids_b, size_b = np.unique(b[b > 0], return_counts=True)
    sizes_a, sizes_b = dict(zip(ids_a.tolist(), size_a.tolist())), dict(zip(ids_b.tolist(), size_b.tolist()))
    both = (a > 0) & (b > 0)
    pairs, overlaps = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    candidates = []
    for (i, j), overlap in zip(pairs.T.tolist(), overlaps.tolist()):
        iou = overlap / (sizes_a[i] + sizes_b[j] - overlap)
        if iou >= 0.5:
            candidates.append((-iou, i, j))
    used_a, used_b, matched = set(), set(), []
    for negative, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append(-negative)
    fraction = len(matched) / len(sizes_a) if sizes_a else 1.0
    median = float(np.median(matched)) if matched else float("nan")
    return len(sizes_a), len(sizes_b), fraction, median


def within_l12_bounds(n_reference, n_other, fraction, median):
    return abs(n_other - n_reference) <= max(1, 0.01 * n_reference) and fraction >= 0.99 and median >= 0.99


def test_the_l12_metrics_on_hand_built_labels():
    """The metric code of L12 on boxes whose IoU is known: 1.0 for equal boxes, 4/6 for a shifted one."""
    reference = np.zeros((1, 8, 12), np.uint32)
    reference[0, 1:3, 1:4], reference[0, 5:7, 5:8] = 1, 2          # 6 pixels each
    assert l12_agreement(reference, reference) == (2, 2, 1.0, 1.0)
    shifted = reference.copy()
    shifted[0, 5:7, 5:8] = 0
    shifted[0, 5:7, 6:9] = 9                                        # 4 of 6 pixels shared: IoU 4/8
    assert l12_agreement(reference, shifted) == (2, 2, 1.0, (1.0 + 0.5) / 2)
    shifted[0, 5:7, 6:9] = 0
    shifted[0, 5:7, 7:10] = 9                                       # 2 shared: IoU 2/10, unmatched
    assert l12_agreement(reference, shifted) == (2, 2, 0.5, 1.0)
    split = reference.copy()
    split[0, 5:7, 5:8] = 0
    split[0, 5:7, 6:7], split[0, 5, 5], split[0, 6, 7] = 3, 4, 5    # 2 + 1 + 1 pixels of label 2
    assert l12_agreement(reference, split) == (2, 4, 0.5, 1.0)
    assert within_l12_bounds(100, 101, 0.99, 0.99) and within_l12_bounds(1, 2, 1.0, 1.0)
    assert not within_l12_bounds(100, 102, 1.0, 1.0) and not within_l12_bounds(100, 100, 0.98, 1.0)
    assert not within_l12_bounds(100, 100, 1.0, 0.989)


L12_CALLS = """
import os, sys
import numpy as np
from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid, SegmentationInput, StarDistConfig, segment
device, parity, out = sys.argv[1:4]
configs = {"P1_ln_dapi_round4_3d": StarDistConfig(model_path=os.environ["STARFINDER_STARDIST_3D_SPLEEN"], scale=1.0),
           "P2_tissue_pi_2d": StarDistConfig(model="2D_versatile_fluo", scale=1.0)}
labels = {}
with np.load(parity) as data:
    for name, config in configs.items():
        image = data[f"input__{name}"]
        image = image if image.ndim == 3 else image[np.newaxis]
        grid = ReferenceGrid(image.shape, ImageMetadata(f"parity_{name}"), "declared")
        result = segment(SegmentationInput(image[..., np.newaxis], grid, ("nuclear",)), config=config,
                         target="nucleus", device=device, label_namespace=f'["parity",null,null,null,"{name}"]')
        assert result.record["methods"][0]["execution"]["device"] == device
        labels[name] = result.labels
np.savez(out, **labels)
"""


def l12_call(python, device, out, env):
    result = subprocess.run([python, "-c", L12_CALLS, device, str(PARITY), str(out)], capture_output=True, text=True,
                            env=env, cwd=TEST_ROOT.parent)
    assert result.returncode == 0, result.stderr[-4000:]
    with np.load(out) as data:
        return {name: data[name] for name in data.files}


@pytest.mark.learned
@pytest.mark.slow
def test_l12_cpu_and_gpu_labels_agree_within_the_w306_bounds(tmp_path):
    gpu_python = os.environ.get(GPU_PYTHON)
    if not gpu_python:
        pytest.skip(f"L12 runs only in a batch whose guidance grants the GPU; {GPU_PYTHON} names the Python of the "
                    "GPU environment and is unset")
    pytest.importorskip("stardist", reason="needs the stardist extra")
    spleen = os.environ.get(SPLEEN_VARIABLE)
    if not spleen or not Path(spleen, "weights_best.h5").is_file():
        pytest.skip(f"the user-trained 3D_spleen model is given by {SPLEEN_VARIABLE}, which is unset or names no "
                    "model folder")
    folder = weights_directory() / "stardist" / "2D_versatile_fluo"
    if not all((folder / f.path).is_file() for f in KNOWN_MODELS[("stardist", "2D_versatile_fluo")].files):
        pytest.skip(f"stardist model 2D_versatile_fluo is not in the weights cache {weights_directory()}")
    assert hashlib.sha256(PARITY.read_bytes()).hexdigest() == PARITY_SHA256
    cpu = l12_call(sys.executable, "cpu", tmp_path / "cpu.npz", one_thread_environment())
    gpu = l12_call(gpu_python, "cuda", tmp_path / "cuda.npz", one_thread_environment(CUDA_VISIBLE_DEVICES="0"))
    assert sorted(cpu) == sorted(gpu) == ["P1_ln_dapi_round4_3d", "P2_tissue_pi_2d"]
    for name in cpu:
        assert cpu[name].shape == gpu[name].shape and cpu[name].dtype == gpu[name].dtype == np.uint32
        metrics = l12_agreement(cpu[name], gpu[name])
        assert within_l12_bounds(*metrics), (name, metrics)
