"""Segmentation models: the known-models table, hash-checked resolution and the explicit fetch.

Models live in the §2.7 weights cache (``STARFINDER_WEIGHTS_DIR``, default
``~/.cache/starfinder/weights``) as ``<root>/<method>/<model>/``; a user-trained
model is given by path. Resolution never uses the network and never calls a
library: it checks that the files exist and recomputes their SHA-256
(docs/segmentation-contract.md, "Models"). Only :func:`_fetch_model`, which the
``starfinder weights fetch`` command calls and no segmentation entry imports,
downloads.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from starfinder.spot_finding._weights import sha256_file, weights_directory

from ._errors import MissingModelError, ModelHashMismatchError

RECORD_NAME = "starfinder-weights.json"
#: The files a user-trained model folder must hold, by method; a Cellpose model is one file.
REQUIRED_FILES = {"stardist": ("config.json", "thresholds.json", "weights_best.h5")}


@dataclass(frozen=True)
class ModelFile:
    """A file the library loads: path relative to the model folder, SHA-256 and size in bytes."""
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class KnownModel:
    """One row of the known-models table (docs/segmentation-contract.md, "Known-models table").

    url is the file ``starfinder weights fetch`` downloads, with its SHA-256 and
    size (None when not recorded); archive is True when the download is a zip
    extracted into the model folder (StarDist) and False when it is the loaded
    file itself (Cellpose). files are the files the library loads, relative to
    the folder ``<root>/<method>/<model>/``. dimensionality, channels, grid,
    stored_thresholds (``prob`` and ``nms``, or None) and training_pixel_size
    are the documented model properties; source names where the table's hashes
    come from.
    """
    method: str
    model: str
    url: str
    sha256: str
    bytes: int | None
    archive: bool
    files: tuple[ModelFile, ...]
    dimensionality: str
    channels: str
    grid: tuple[int, ...] | None
    stored_thresholds: Mapping[str, float] | None
    training_pixel_size: str
    source: str


#: The known-models table: (method, model) -> KnownModel. Pretrained models Starfinder can verify; it sets no
#: default model. User-trained models such as 3D_spleen are given by path and are not listed.
KNOWN_MODELS: dict[tuple[str, str], KnownModel] = {
    (entry.method, entry.model): entry for entry in (
        KnownModel(
            "stardist", "2D_versatile_fluo",
            "https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip",
            "4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd", 5320433, True, (
                ModelFile("config.json", "836da16282c3e0db1ba2e58f377e977419887ace8c737ebde16a094d58d50f74", 1021),
                ModelFile("thresholds.json", "5cd6aac6e923f8659b63a9e297485920d45a59dad35371ddb8b6ae4be398d805", 39),
                ModelFile("weights_best.h5", "42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349",
                          5771480)),
            "2D YX", "1", (2, 2), {"prob": 0.479071463157368, "nms": 0.3},
            "not documented (fluorescent nuclei, DSB 2018 subset); train patch 256x256",
            "stardist-models release v0.1; archive SHA-256 as StarDist 0.9.2 registers it (W-306 models.csv)"),
        KnownModel(
            "cellpose", "cpsam_v2", "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2",
            "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667", 1233586851, False, (
                ModelFile("cpsam_v2", "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
                          1233586851),),
            "2D network; 3D by planes (do_3d)", "1 to 3", None, None,
            "none: images are rescaled to a 30-pixel diameter",
            "the Cellpose 4.2.1.1 URL, fetched in W-305; no published hash was checked (W-306 notes section 8)"),
    )
}


def known_model(method: str, model: str) -> KnownModel:
    """The table entry of (method, model); ValueError lists the known models of the method."""
    entry = KNOWN_MODELS.get((method, model))
    if entry is None:
        known = sorted(m for k, m in KNOWN_MODELS if k == method)
        raise ValueError(f"unknown {method} model {model!r}; known models: {known} (a user-trained model is given "
                         "by model_path)")
    return entry


def fetch_command(method: str, model: str, directory=None) -> str:
    return f"starfinder weights fetch {method} {model}" + (f" --dir {directory}" if directory is not None else "")


def _check_expected(model_sha256):
    expected = dict(model_sha256 or {})
    for name, digest in expected.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("model_sha256 maps file names to 64-character SHA-256 hex digests")
    return expected


def resolve_model(method: str, *, model: str | None = None, model_path=None,
                  model_sha256: Mapping[str, str] | None = None, directory=None) -> tuple[Path, list[dict]]:
    """Verified local model of a segmentation method, without the network and without the library.

    A known model (``model``) is looked up in :data:`KNOWN_MODELS` and resolved in
    the weights cache, ``<root>/<method>/<model>/``: every file the table lists
    must exist and have the table's SHA-256. A user-trained model
    (``model_path``) must exist; a StarDist folder must hold ``config.json``,
    ``thresholds.json`` and ``weights_best.h5`` (only these are hashed), a
    Cellpose model is the file itself, and a folder of another method is hashed
    file by file. ``model_sha256`` maps file names to expected SHA-256 values
    and is checked for a path. Nothing is downloaded.

    Parameters
    ----------
    method : str
        The segmentation method name (``stardist``, ``cellpose``).
    model, model_path
        Exactly one: a known model name, or the folder or file of a
        user-trained model.
    model_sha256 : Mapping[str, str] | None
        Expected SHA-256 per file name, for ``model_path``.
    directory : path-like | None
        Root of the weights cache; default as ``weights_directory``.

    Returns
    -------
    tuple[pathlib.Path, list[dict]]
        The absolute model path (the folder; for Cellpose the file) and one
        provenance ``artifacts`` entry per hashed file (``name``, ``path``,
        ``sha256``, ``source``: ``"cache"`` with the table's ``url``, or
        ``"path"``).

    Raises
    ------
    ValueError
        Neither or both of model and model_path, an unknown model name, or a
        malformed model_sha256.
    MissingModelError
        The folder or a needed file is missing (naming the method, the model or
        path and, for a known model, the fetch command).
    ModelHashMismatchError
        A file's SHA-256 differs from the table or from model_sha256 (naming
        the file and both hashes).
    """
    if (model is None) == (model_path is None):
        raise ValueError(f"{method} needs exactly one of model and model_path")
    if model is not None:
        entry = known_model(method, model)
        folder = weights_directory(directory) / method / model
        artifacts = []
        for item in entry.files:
            path = folder / item.path
            if not path.is_file():
                raise MissingModelError(f"{method} model {model!r}: file {path} is missing; segmentation never "
                                        f"downloads a model; fetch it with '{fetch_command(method, model, directory)}'")
            digest = sha256_file(path)
            if digest != item.sha256:
                raise ModelHashMismatchError(f"{method} model {model!r}: {path} has SHA-256 {digest}, expected "
                                             f"{item.sha256}")
            artifacts.append({"name": f"{method}/{model}/{item.path}", "path": str(path), "sha256": digest,
                              "source": "cache", "url": entry.url})
        resolved = folder / entry.files[0].path if not entry.archive else folder
        return resolved, artifacts

    expected = _check_expected(model_sha256)
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise MissingModelError(f"segmentation method {method!r}: model path {path} does not exist; segmentation "
                                "never downloads a model")
    if method == "cellpose" and not path.is_file():
        raise MissingModelError(f"segmentation method 'cellpose': model path {path} is not a file")
    required = REQUIRED_FILES.get(method, ())
    if required and not path.is_dir():
        raise MissingModelError(f"segmentation method {method!r}: model path {path} is not a folder")
    if path.is_dir():
        files = [path / name for name in required] if required else sorted(p for p in path.iterdir() if p.is_file())
        files += [path / name for name in sorted(expected) if path / name not in files]
    else:
        files = [path] + [path.parent / name for name in sorted(expected) if name != path.name]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise MissingModelError(f"segmentation method {method!r}: model {path} has no {missing}")
    artifacts = []
    for file in files:
        digest = sha256_file(file)
        if file.name in expected and expected[file.name] != digest:
            raise ModelHashMismatchError(f"{file}: SHA-256 {digest} differs from the expected {expected[file.name]}")
        artifacts.append({"name": f"{method}/{path.name}/{file.name}" if path.is_dir() else f"{method}/{path.name}",
                          "path": str(file), "sha256": digest, "source": "path"})
    return path, artifacts


def model_dimensions(path: Path, config) -> int | None:
    """The model's dimensionality without the library: a config's ``do_3d`` (Cellpose), else config.json's ``n_dim``."""
    if hasattr(config, "do_3d"):
        return 3 if config.do_3d else 2
    if path.is_dir() and (path / "config.json").is_file():
        return json.loads((path / "config.json").read_text()).get("n_dim")
    return None


# --- The explicit fetch (starfinder weights fetch); no segmentation entry imports these -------------------

def _verify_folder(entry, folder):
    """Missing listed files of a present folder; a present file that differs raises (it is never overwritten)."""
    missing = []
    for item in entry.files:
        path = folder / item.path
        if not path.is_file():
            missing.append(item)
            continue
        digest = sha256_file(path)
        if digest != item.sha256:
            raise ModelHashMismatchError(
                f"{entry.method} model {entry.model!r}: {path} has SHA-256 {digest}, expected {item.sha256}; "
                f"fetch never overwrites a present file, so remove {folder} and fetch again")
    return missing


def _fetch_model(method: str, model: str, *, directory=None) -> Path:
    """Download, verify and install one known segmentation model into the cache; return its folder.

    The download is checked against the table's size and SHA-256, extracted
    (StarDist) or moved (Cellpose) and each listed file is checked before it is
    moved into ``<root>/<method>/<model>/`` next to a ``starfinder-weights.json``
    record. A verified present file is never replaced; a present file that
    differs raises ModelHashMismatchError. This is the only segmentation code
    that uses the network.
    """
    from starfinder.spot_finding._fetch import _download, _extract

    entry = known_model(method, model)
    root = weights_directory(directory)
    parent, folder = root / method, root / method / model
    missing = _verify_folder(entry, folder) if folder.exists() else list(entry.files)
    if not missing:
        return folder
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{model}.", suffix=".tmp", dir=parent))
    try:
        download = staging / "download"
        size, digest, _ = _download(entry.url, download)
        if digest != entry.sha256 or (entry.bytes is not None and size != entry.bytes):
            raise ModelHashMismatchError(f"{entry.url}: downloaded {size} bytes with SHA-256 {digest}, expected "
                                         f"{entry.bytes} bytes with SHA-256 {entry.sha256}")
        if entry.archive:
            extracted = _extract(download, staging / "extracted")
        else:
            extracted = staging / "extracted"
            extracted.mkdir()
            os.replace(download, extracted / entry.files[0].path)
        for item in entry.files:
            path = extracted / item.path
            if not path.is_file() or sha256_file(path) != item.sha256:
                raise ModelHashMismatchError(f"{entry.url}: {item.path} is missing from the download or differs "
                                             f"from the expected SHA-256 {item.sha256}")
        folder.mkdir(parents=True, exist_ok=True)
        for item in missing:
            os.replace(extracted / item.path, folder / item.path)
        record = {"method": method, "model": model, "url": entry.url, "sha256": entry.sha256, "bytes": entry.bytes,
                  "fetched_at": datetime.now(timezone.utc).isoformat(),
                  "files": {item.path: item.sha256 for item in entry.files}}
        (staging / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
        os.replace(staging / RECORD_NAME, folder / RECORD_NAME)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    resolve_model(method, model=model, directory=directory)
    return folder
