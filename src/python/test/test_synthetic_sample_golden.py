"""Golden digests of the saved sample presets (docs/synthetic-cellular-scene.md, "Opt-in").

The richer cellular scene is opt-in: a sample scene without its settings is exactly
today's. This file pins what ``save_sample_scene`` writes for ``sample-tiny-v1`` and
``sample-small-v1`` as generated at 943bb6b, so that a change of any generated value of
an existing preset fails here. Digests are of content, by group of files:

* CSV files as written;
* TIFF files as their pixel arrays (dtype, shape and bytes), because the file bytes
  also hold the writer's version string;
* the JSON records (``sample.json``, each ``provenance.json``) as canonical JSON without
  ``numpy_version`` and without the ``files`` map of ``sample.json`` (the SHA-256 of the
  TIFF files as written); the array digests they hold stay in.

Repeatability is that of the formed model (docs/synthetic-specification.md, "Random
streams"): exact for a pinned NumPy build on one CPU. NumPy's vectorised functions differ
in the last bit between CPUs, so the floating-point values of the CSV files and the JSON
records are compared to 6 significant digits, with magnitudes below 1e-9 taken as 0
(amendment of 2026-10-19 to GN1: a GitHub runner's CSV files differed from the lab
machine's only there). Integers, strings and pixel arrays stay exact.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene

pytestmark = [pytest.mark.synthetic, pytest.mark.golden]

# Generated at 943bb6b in dev-s210-export-20261007T050852Z (NumPy 2.2.6), one CPU, one thread. Taken again on
# 2026-10-19 at 8606450 with the floats to 6 significant digits (GN1's amendment of 2026-10-19): there every group
# still equalled the exact pins of 943bb6b on the lab machine, and the new pins are the same with NumPy's AVX-512
# paths turned off.
PINNED = {
    "sample-tiny-v1": {
        "positions.csv": "7fb4663d4ba10aded41fe94227712d7279ab9ff8031894404f78e7ed9b228d43",
        "placement_truth.csv": "98aa6ce8558d9e8a0e9d5e87ce77721b63266617311d30c14bcf77e566897a93",
        "nuclei.csv": "dbef41063148f34ab7a7bfd708f77ce23db76af6a5653831f1fa76c82575c126",
        "cells.csv": "e9fbac588824e523ab8c17ee35d566557d8976d6402100b08e335c26fb151e02",
        "molecules.csv": "82ee4e3b79a37dd17b047de9550a3d57090ed11f2a7bd722df3c17d0732a4e20",
        "cell_fovs.csv": "cc3a9ae57746ee9ef2d8693ba2d92b40a87b51d96bf3ea935a819ec4eb013a63",
        "molecule_fovs.csv": "8f027049b77080e514882b3cb0f455b7f03d68d4e0830142c4483dda10d88dc2",
        "sample labels": "592d2eadb0a2bec2fb69594ba0fd505737729cba49f69cfd1a3185e2609e81ad",
        "fov images": "6ce3926cb9e0f810861f56fa621c3484ab63ad6aa7b579d7c97b12e3984d099b",
        "fov stains": "5579572204f3808e9046640b3b94335bc96bdf096e1993c15f6399427533975d",
        "fov labels": "ace7448e66c1d78552b2c6da57ddee52d8361bfd294782435f475e6914b34105",
        "fov tables": "729ba3607ceb09597e3470b43d37b7768eceb5f981f239b4462ebedcaf82e7ed",
        "records": "9c57dda47387a642977f6142dad613ed0b89cb012ef658705c6664c2a1dac90e",
    },
    "sample-small-v1": {
        "positions.csv": "45e5e7c5b334ed6ec6e0a861ac0bba858ccf02b7a5485a88ec5438be9e691e3e",
        "placement_truth.csv": "b92171961b9c0dea2b25e152eac8ef632b2f51109efbdaf69c4f2787b7cee092",
        "nuclei.csv": "353ce9aa2d6a608768ce7400eef05da31111e14e57189ee78664b13686062447",
        "cells.csv": "e0f7c9f9629f4c02f903b100e1e58d89283ac9849bae18ede962a1be6f5ab706",
        "molecules.csv": "eb901673814b72b2d43d89cab0eff4368f9df282e01c0302b6a82f8518612432",
        "cell_fovs.csv": "824e442163fbc5550e587f2e911265ef480cfb9c35b4ef15f0c71542c1990d08",
        "molecule_fovs.csv": "a0e1ba8f3ad822f1a685eda24acada00e0453a3a461f3f2f1d66c37425618f1f",
        "sample labels": "066220555d7977b0d051ab963f4ae06e641b9d8a9446ad824e07e712e98d4fd8",
        "fov images": "ff6fd38c7ca95d13e30791cf36009c804fa0e3ef8e7f541c52efe59fb5265d1d",
        "fov stains": "2043a01700b1ffa80c098b728a7bca4e5fd46cb8686af38c46215bef6f82b54f",
        "fov labels": "525b45f163bf54d79bfc14f58176a04e92f8dd60a61efafdb55879a932a84ea3",
        "fov tables": "637c823e6eff98f92944f042f4dbd96ad42122dcd51dce44295d41b4593baf39",
        "records": "0280c4bf644bb933e4b275806a9f86d6196ba408eec6e5053f166aa6e714354d",
    },
}
GROUPS = ("positions.csv", "placement_truth.csv", "nuclei.csv", "cells.csv", "molecules.csv", "cell_fovs.csv",
          "molecule_fovs.csv", "sample labels", "fov images", "fov stains", "fov labels", "fov tables", "records")


def _number(value):
    """A float to 6 significant digits; magnitudes below 1e-9 are 0 (the last bit differs between CPUs)."""
    return 0.0 if abs(value) < 1e-9 else float(f"{value:.6g}")


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k != "numpy_version"}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return _number(value)
    return value


def _cell(text):
    if not any(c in text for c in ".eE"):
        return text
    try:
        return repr(_number(float(text)))
    except ValueError:
        return text


def _csv_payload(path):
    rows = csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline=""))
    return "\n".join("\x1f".join(_cell(cell) for cell in row) for row in rows).encode()


def _file_digest(root, name):
    path = root / name
    if name.endswith(".tif"):
        array = np.ascontiguousarray(tifffile.imread(path))
        payload = f"{array.dtype.str}{array.shape}".encode() + array.tobytes()
    elif name.endswith(".json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if name == "sample.json":
            record.pop("files")
        payload = json.dumps(_canonical(record), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    elif name.endswith(".csv"):
        payload = _csv_payload(path)
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _group(name):
    if not name.startswith("fovs/"):
        return "sample labels" if name.startswith("labels/") else "records" if name == "sample.json" else name
    if "/images/" in name:
        return "fov images"
    if "/labels/" in name:
        return "fov labels"
    if name.endswith("stain.ome.tif"):
        return "fov stains"
    return "records" if name.endswith("provenance.json") else "fov tables"


def saved_digests(root: Path) -> dict[str, str]:
    """One SHA-256 per group, over the sorted (file, content digest) pairs of that group."""
    members = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = path.relative_to(root).as_posix()
        members.setdefault(_group(name), []).append(f"{name}\0{_file_digest(root, name)}\n")
    return {group: hashlib.sha256("".join(lines).encode()).hexdigest() for group, lines in members.items()}


@pytest.mark.parametrize("name", [
    "sample-tiny-v1",
    pytest.param("sample-small-v1", marks=pytest.mark.slow),
])
def test_saved_preset_outputs_equal_the_pins(tmp_path, name):
    book, config = sample_scene_preset(name)
    root = save_sample_scene(generate_sample_scene(book, config=config), tmp_path / name)
    digests = saved_digests(root)
    assert sorted(digests) == sorted(GROUPS)
    changed = {group: digests[group] for group in GROUPS if digests[group] != PINNED[name].get(group)}
    assert not changed, f"{name}: groups whose saved content differs from the pins: {changed}"
