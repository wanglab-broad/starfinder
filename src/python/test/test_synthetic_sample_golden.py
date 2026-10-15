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
streams"): exact for a pinned NumPy build on one CPU.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene

pytestmark = [pytest.mark.synthetic, pytest.mark.golden]

# Generated at 943bb6b in dev-s210-export-20261007T050852Z (NumPy 2.2.6), one CPU, one thread.
PINNED = {
    "sample-tiny-v1": {
        "positions.csv": "6d5cda005ccd3324dbd42c21a6ba43ad6fbfc5aaa5f5338f16bb8972ab19785a",
        "placement_truth.csv": "e11d5a38bcdaa05124884773fd9e853ac93e48359e3a8bc8b6a8b4eb31428f0b",
        "nuclei.csv": "c0132ff73481805d58d24d59661902080807ddfef86dd53d403255a4a5c95f8e",
        "cells.csv": "bab3476c3533c0b96131a82054708694789aaf17f01880ba6c726d848053256f",
        "molecules.csv": "0626d1066b25ee1e27ede459167ef414d7cbcc11194ca2664c1eb5017bba3c77",
        "cell_fovs.csv": "d2dfd62fec40bfa08bbdac1e493180ec7acc51acf67599fbc1199d8f8da87689",
        "molecule_fovs.csv": "11a0cd7ee9e1e95f743d38b5facb51258a7741c360039e0cf9edbb9f595ead0e",
        "sample labels": "592d2eadb0a2bec2fb69594ba0fd505737729cba49f69cfd1a3185e2609e81ad",
        "fov images": "6ce3926cb9e0f810861f56fa621c3484ab63ad6aa7b579d7c97b12e3984d099b",
        "fov stains": "5579572204f3808e9046640b3b94335bc96bdf096e1993c15f6399427533975d",
        "fov labels": "ace7448e66c1d78552b2c6da57ddee52d8361bfd294782435f475e6914b34105",
        "fov tables": "6c04a446cd4e90e5c6cff66b5d5719e2416632979dd2ce2afe5bed4de136e7cd",
        "records": "be27331a45bbfad9ed496b29cb7ec1a235e6323dc483ae3b79aba7d4527af524",
    },
    "sample-small-v1": {
        "positions.csv": "628e8a6a2afef365f3e6d9aaaf6f9fed54f9853302271230c67e39947d4287bc",
        "placement_truth.csv": "05a12497456025d499406dcbca38399df59869d61f21ba93b66afe56e755ff38",
        "nuclei.csv": "a2ba391a510da0c19b6f8ca4ea3d6118800e1d4cde6ed8057b6f8ecc45ba9d1d",
        "cells.csv": "931dd6045da02d89b18c9ad1af868b6048be117c09758a0aac8f8dda9ff0abe9",
        "molecules.csv": "29abd3522d334c4ccee05658b3f943287fc685849cfa226f1f4d3b835aa5f631",
        "cell_fovs.csv": "5233396f9c4aadcd2d905225266ad3f79871bb7ee83deab9c672b56621300cb5",
        "molecule_fovs.csv": "8b670f3a5953626b675eecf93546b4260187ce86f33a59d1b2844d355e3a2c2f",
        "sample labels": "066220555d7977b0d051ab963f4ae06e641b9d8a9446ad824e07e712e98d4fd8",
        "fov images": "ff6fd38c7ca95d13e30791cf36009c804fa0e3ef8e7f541c52efe59fb5265d1d",
        "fov stains": "2043a01700b1ffa80c098b728a7bca4e5fd46cb8686af38c46215bef6f82b54f",
        "fov labels": "525b45f163bf54d79bfc14f58176a04e92f8dd60a61efafdb55879a932a84ea3",
        "fov tables": "752bb56d48b5c6656a04e26fe2bece149829975a84568ea7a06c730ec30fcc46",
        "records": "2c55a0b9a6aba93323dc0e6cea46d11744d7e32bb27526dd2e05c13adf67417c",
    },
}
GROUPS = ("positions.csv", "placement_truth.csv", "nuclei.csv", "cells.csv", "molecules.csv", "cell_fovs.csv",
          "molecule_fovs.csv", "sample labels", "fov images", "fov stains", "fov labels", "fov tables", "records")


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k != "numpy_version"}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


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
