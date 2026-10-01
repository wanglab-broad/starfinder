"""Pretrained weights: the known-weights table, the local cache and hash-checked resolution.

Starfinder fetches weights only on an explicit command (``starfinder weights
fetch`` or fetch_weights) and never through a detection: this module does not
use the network, and the fetch code lives in a module that detection never
imports. Weights are loaded from explicit paths in the cache, never from the
libraries' own caches.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from ._errors import MissingWeightsError, WeightsHashMismatchError

RECORD_NAME = "starfinder-weights.json"
_SPOTIFLOW_RELEASE = "https://github.com/weigertlab/spotiflow-models/releases/download/0.6.0"
_PISCIS_REVISION = "9bdefc72cb519053c63fd2d7bff9d12db7bb394e"
_PISCIS_FILES = f"https://huggingface.co/wniu/Piscis/resolve/{_PISCIS_REVISION}/models"
_SPOTIFLOW_PIXELS = "https://weigertlab.org/spotiflow/pretrained.html"
_OPERATOR = ("operator-retrieved on 2026-10-01 (W-266 recovery note R-20261001T005236Z-898a772a); "
             "not verified by a worker")


@dataclass(frozen=True)
class WeightsFile:
    """A file of a model folder that the library loads: path relative to the folder, SHA-256 and size."""
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class KnownWeights:
    """One row of the known-weights table (W-266 known-weights.csv).

    url is the file Starfinder downloads, with its SHA-256 and size; md5 is
    the checksum the library registers for it (Spotiflow archives), or None.
    archive is True when the download is a zip extracted into the model
    folder (Spotiflow) and False when it is moved there as is (Piscis).
    files are the files the library loads, relative to the model folder.
    min_shape_zyx is the observed minimum input shape (a Z of 1 means plane
    input). native_threshold is the threshold stored with the weights or the
    library default. extracted is every file a Spotiflow archive extracts to,
    with its SHA-256 and size, recorded from the verified archives (empty for
    Piscis, whose download is the loaded file); resolve_weights checks the ones
    a caller names.
    """
    method: str
    model: str
    dimensionality: str
    url: str
    revision: str
    sha256: str
    bytes: int
    md5: str | None
    archive: bool
    files: tuple[WeightsFile, ...]
    min_shape_zyx: tuple[int, int, int]
    training_pixel_size: str
    training_pixel_size_provenance: str
    native_threshold: float
    extracted: tuple[WeightsFile, ...] = ()


def _spotiflow(model, dimensionality, sha256, size, md5, best, best_bytes, minimum, pixels, threshold, extracted):
    best = WeightsFile("best.pt", best, best_bytes)
    extracted = tuple(sorted((best, *(WeightsFile(*item) for item in extracted)), key=lambda f: f.path))
    return KnownWeights(
        "spotiflow", model, dimensionality, f"{_SPOTIFLOW_RELEASE}/{model}.zip", "spotiflow-models release 0.6.0",
        sha256, size, md5, True, (best,), minimum, pixels, f"{_SPOTIFLOW_PIXELS}; {_OPERATOR}", threshold, extracted)


def _piscis(model, sha256, size):
    return KnownWeights(
        "piscis", model, "2D network; stack mode links planes", f"{_PISCIS_FILES}/{model}.pt",
        f"wniu/Piscis {_PISCIS_REVISION}", sha256, size, None, False, (WeightsFile(f"{model}.pt", sha256, size),),
        (2, 1, 1), "not published by the authors", _OPERATOR, 0.5)


# The known-weights table: (method, model) -> KnownWeights. It sets no default model. The extracted files of
# the Spotiflow archives besides best.pt were hashed from the archives that matched the table's SHA-256 and the
# library's MD5 (W-272; the operator's fetched copies and the W-266 spike copies agree).
KNOWN_WEIGHTS: dict[tuple[str, str], KnownWeights] = {
    (entry.method, entry.model): entry for entry in (
        _spotiflow("synth_3d", "3D", "2468125c8c1a8bb6f6364f439d50ff08e24a15e6fa4c0c2cf43b26667393038b", 263107693,
                   "a031f1284590886fbae37dc583c0270d",
                   "846d1ef438f872d50f160ad6cfb8be5b99f3f837600c52843a1391ca4eae7d4c", 142064642, (7, 8, 8),
                   "0.2 um voxels (synthetic)", 0.3, (
                       ("config.yaml", "09bd576178014753853c4dab7066fd300b3ed539e053ddaa1b4192f89a00131d", 475),
                       ("last.pt", "11a0ef1b3dc11c9d0a9a497bed301cef24f2e43c740b02f8db74fadd75cd9227", 142064642),
                       ("thresholds.yaml", "f63b656f20f5c45143686c34d04f20a08c078f6e5b88ad2e0f40aab0de24f04f", 48),
                       ("train_config.yaml", "37909c8acf90b827b33a27daf9d0c08ea2af2ce297e0b7ad4d494a04a20234c0",
                        276))),
        _spotiflow("smfish_3d", "3D", "a6c79f767eaddff7e4c85ba370b370ffdb752829ea0a06d7ca71e5556b554775", 263106200,
                   "c5ab30ba3b9ccb07b4c34442d1b5b615",
                   "1fdfd62c89a007094870782c27da163052ed72952a72d29f12e6730514d3ad6d", 142064642, (7, 8, 8),
                   "0.13 um YX, 0.48 um Z", 0.4, (
                       ("config.yaml", "09bd576178014753853c4dab7066fd300b3ed539e053ddaa1b4192f89a00131d", 475),
                       ("last.pt", "bf1d5f13928c09a4b03c80fed41fd292a150183c28f4926a810688bd82dc4a18", 142064642),
                       ("thresholds.yaml", "daa3d82d9bc79141e904bcab6ea2f24057b966b7ad43b8053383d4d7f7c4e6e5", 48),
                       ("train_config.yaml", "37909c8acf90b827b33a27daf9d0c08ea2af2ce297e0b7ad4d494a04a20234c0",
                        276))),
        _spotiflow("general", "2D", "1da93a8282fedab697dbb9f5c24f623e452063b296ffaad9becd0afdc7bd2797", 87885382,
                   "9dd31a36b737204e91b040515e3d899e",
                   "1c3575464d621924b27f4deb66495b807f175a0ccd995d3533403f00daf806f2", 47408604, (1, 6, 6),
                   "0.04, 0.1, 0.11, 0.15, 0.32 and 0.34 um (mixed training data)", 0.49999999999999994, (
                       ("config.yaml", "507b60fab22c08e8819da3f7c84d8a85b3fdce4caca14c18d101b502109b0649", 389),
                       ("last.pt", "2d1c04545f2205de46d7002c460d492cebe0a5c76840b494668b3ede904651fc", 47408604),
                       ("thresholds.yaml", "4c054d3cbb2825df7a4672a63ef454fbc25a7a01e53faefc431793068ac6367e", 76),
                       ("train_config.yaml", "fa805d53d763da34b68ddbaf6fd6876ff0d83d7565c4280f8b87ee14b7cd9777",
                        257))),
        _spotiflow("hybiss", "2D", "d6221339a104cddfac77b3b8a1014bd316a7060934d69bc078f5a653cb72658b", 87945684,
                   "254afa97c137d0bd74fd9c1827f0e323",
                   "fa5d5cb313bcfd75527f3d0962ccc3654bc7aa33359097d16f3b0835b0103bd1", 47408604, (1, 6, 6),
                   "0.15, 0.32 and 0.34 um", 0.5319999999999999, (
                       ("config.yaml", "10253c542bcc21348a1a5b3810de1950be7250be51147f96a8c679eef3cc583a", 368),
                       ("last.pt", "2b309de99caf9a948b922fa88e99b5599b335b77b246fd0f5d4ea2ec55f89632", 47408604),
                       ("thresholds.yaml", "62dee75197686a9ea85fe0bf493f8ec2bcefbea20ef2748948d437b627e44045", 74),
                       ("train_config.yaml", "480f56279201137dc1ec74681dfd160b2d98af12d7569b8fdaad667e45bbc7cd",
                        244))),
        _piscis("20230905", "57177963f50af929d68c8a2e5797bb966cb7bc43806f8c1efe9f7131b6449f19", 30077822),
        _piscis("20251212", "e4ec9fe68e43fe955001e3bf2317badfc4c418027910e7762a37f52d7e64d06f", 30143014),
    )
}


def known_weights(method: str, model: str) -> KnownWeights:
    """The table entry of (method, model); ValueError lists the known models of the method."""
    entry = KNOWN_WEIGHTS.get((method, model))
    if entry is None:
        known = sorted(m for k, m in KNOWN_WEIGHTS if k == method)
        raise ValueError(f"unknown {method} model {model!r}; known models: {known}" if known else
                         f"no known weights for method {method!r}; known methods: "
                         f"{sorted({k for k, _ in KNOWN_WEIGHTS})}")
    return entry


def weights_directory(directory=None) -> Path:
    """Root of the weights cache.

    directory when given; otherwise STARFINDER_WEIGHTS_DIR when set, else
    $XDG_CACHE_HOME/starfinder/weights (default ~/.cache/starfinder/weights).
    A model lives in <root>/<method>/<model>/.
    """
    if directory is not None:
        return Path(directory)
    if os.environ.get("STARFINDER_WEIGHTS_DIR"):
        return Path(os.environ["STARFINDER_WEIGHTS_DIR"])
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "starfinder" / "weights"


def model_folder(method: str, model: str, directory=None) -> Path:
    """The folder of one model in the cache (it may not exist)."""
    entry = known_weights(method, model)
    return weights_directory(directory) / entry.method / entry.model


def sha256_file(path, chunk=1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch_command(method, model, directory=None):
    return f"starfinder weights fetch {method} {model}" + (f" --dir {directory}" if directory is not None else "")


def _listed(entry, extracted):
    """The files of entry to verify: its files, then the named extracted files that are not among them."""
    known = {item.path: item for item in entry.extracted}
    unknown = [name for name in extracted if name not in known]
    if unknown:
        raise ValueError(f"{entry.method} model {entry.model!r} extracts no {unknown}; its files are {sorted(known)}")
    loaded = {item.path for item in entry.files}
    return entry.files + tuple(known[name] for name in extracted if name not in loaded)


def resolve_weights(method: str, model: str, *, directory=None, extracted: tuple[str, ...] = ()) -> Path:
    """Verified local folder of a known model, without using the network or the library.

    Checks that the folder and every file the table lists exist, and
    recomputes each file's SHA-256; extracted names further files of the
    entry's extracted list (paths in the folder) to check the same way, such
    as the configuration files Spotiflow reads next to best.pt. Raises
    ValueError for an unknown model or extracted file, MissingWeightsError
    (naming the method, model, expected path and fetch command) when
    something is missing, and WeightsHashMismatchError (naming the file and
    both hashes) when a hash differs.
    """
    entry = known_weights(method, model)
    folder = model_folder(method, model, directory)
    for item in _listed(entry, extracted):
        path = folder / item.path
        if not path.is_file():
            raise MissingWeightsError(
                f"{method} model {model!r}: weights file {path} is missing; fetch it with "
                f"'{fetch_command(method, model, directory)}'")
        actual = sha256_file(path)
        if actual != item.sha256:
            raise WeightsHashMismatchError(
                f"{method} model {model!r}: {path} has SHA-256 {actual}, expected {item.sha256}")
    return folder


def weights_artifacts(method: str, model: str, folder, extracted: tuple[str, ...] = ()) -> list[dict]:
    """The provenance artifacts entries of a resolved model: one per verified file (see resolve_weights)."""
    entry = known_weights(method, model)
    return [{"name": f"{method}/{model}", "path": str((Path(folder) / item.path).resolve()), "sha256": item.sha256,
             "source": entry.url, "revision": entry.revision} for item in _listed(entry, extracted)]


def fetch_weights(method: str, model: str, *, directory=None) -> Path:
    """Download, verify and install one known model into the cache; return its folder.

    The source file is downloaded to a temporary name in the target
    directory, its size, SHA-256 and (Spotiflow) library MD5 are checked,
    and only then is it extracted (Spotiflow) or moved (Piscis) into
    <root>/<method>/<model>/ next to a starfinder-weights.json record of the
    entry and the per-file hashes. A verified copy is never overwritten: it
    is returned as is. This is the only function that uses the network;
    detection never calls it.
    """
    from ._fetch import fetch
    return fetch(known_weights(method, model), weights_directory(directory))
