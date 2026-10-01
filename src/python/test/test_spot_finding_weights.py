"""The known-weights table, the weights cache, fetch, verification, their errors and the CLI group
(W-270, docs/spot-finding-contract.md, "Pretrained weights").

No test uses the network: fixture entries inserted into KNOWN_WEIGHTS download a few-byte file
from a temporary directory through a file:// URL, and socket connections are blocked.
"""
import hashlib
import json
import socket
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from starfinder.__main__ import main
from starfinder.spot_finding import (KNOWN_WEIGHTS, KnownWeights, MissingWeightsError, WeightsFile,
    WeightsHashMismatchError, fetch_weights, resolve_weights)
from starfinder.spot_finding._weights import RECORD_NAME, weights_artifacts, weights_directory

# W-266 known-weights.csv: (method, model) -> (downloaded file SHA-256, bytes, library MD5,
# loaded file, its SHA-256, its bytes), copied from the run directory
# /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-266/20260930T225252Z-967e52bd.
W266 = {
    ("spotiflow", "synth_3d"): (
        "2468125c8c1a8bb6f6364f439d50ff08e24a15e6fa4c0c2cf43b26667393038b", 263107693,
        "a031f1284590886fbae37dc583c0270d", "best.pt",
        "846d1ef438f872d50f160ad6cfb8be5b99f3f837600c52843a1391ca4eae7d4c", 142064642),
    ("spotiflow", "smfish_3d"): (
        "a6c79f767eaddff7e4c85ba370b370ffdb752829ea0a06d7ca71e5556b554775", 263106200,
        "c5ab30ba3b9ccb07b4c34442d1b5b615", "best.pt",
        "1fdfd62c89a007094870782c27da163052ed72952a72d29f12e6730514d3ad6d", 142064642),
    ("spotiflow", "general"): (
        "1da93a8282fedab697dbb9f5c24f623e452063b296ffaad9becd0afdc7bd2797", 87885382,
        "9dd31a36b737204e91b040515e3d899e", "best.pt",
        "1c3575464d621924b27f4deb66495b807f175a0ccd995d3533403f00daf806f2", 47408604),
    ("spotiflow", "hybiss"): (
        "d6221339a104cddfac77b3b8a1014bd316a7060934d69bc078f5a653cb72658b", 87945684,
        "254afa97c137d0bd74fd9c1827f0e323", "best.pt",
        "fa5d5cb313bcfd75527f3d0962ccc3654bc7aa33359097d16f3b0835b0103bd1", 47408604),
    ("piscis", "20230905"): (
        "57177963f50af929d68c8a2e5797bb966cb7bc43806f8c1efe9f7131b6449f19", 30077822, None, "20230905.pt",
        "57177963f50af929d68c8a2e5797bb966cb7bc43806f8c1efe9f7131b6449f19", 30077822),
    ("piscis", "20251212"): (
        "e4ec9fe68e43fe955001e3bf2317badfc4c418027910e7762a37f52d7e64d06f", 30143014, None, "20251212.pt",
        "e4ec9fe68e43fe955001e3bf2317badfc4c418027910e7762a37f52d7e64d06f", 30143014),
}
PAYLOAD = b"starfinder fixture weights\n"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a weights test tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(root))
    return root


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def piscis_fixture(tmp_path, monkeypatch):
    """A single-file fixture model: the downloaded file is the file the library loads."""
    source = tmp_path / "source" / "fixture.pt"
    source.parent.mkdir()
    source.write_bytes(PAYLOAD)
    entry = replace(KNOWN_WEIGHTS[("piscis", "20251212")], model="fixture", url=source.as_uri(),
                    revision="fixture", sha256=sha256(PAYLOAD), bytes=len(PAYLOAD),
                    files=(WeightsFile("fixture.pt", sha256(PAYLOAD), len(PAYLOAD)),))
    monkeypatch.setitem(KNOWN_WEIGHTS, ("piscis", "fixture"), entry)
    return entry


@pytest.fixture
def archive_fixture(tmp_path, monkeypatch):
    """A zip fixture model with a top folder and macOS metadata, as the Spotiflow archives have."""
    archive = tmp_path / "source" / "fixture_3d.zip"
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("fixture_3d/best.pt", PAYLOAD)
        handle.writestr("fixture_3d/config.yaml", b"is_3d: true\n")
        handle.writestr("__MACOSX/fixture_3d/._best.pt", b"x")
    data = archive.read_bytes()
    entry = replace(KNOWN_WEIGHTS[("spotiflow", "smfish_3d")], model="fixture_3d", url=archive.as_uri(),
                    revision="fixture", sha256=sha256(data), bytes=len(data), md5=hashlib.md5(data).hexdigest(),
                    files=(WeightsFile("best.pt", sha256(PAYLOAD), len(PAYLOAD)),))
    monkeypatch.setitem(KNOWN_WEIGHTS, ("spotiflow", "fixture_3d"), entry)
    return entry


# --- The table ------------------------------------------------------------------------------------------

def test_known_weights_holds_the_six_w266_rows():
    table = {key: (e.sha256, e.bytes, e.md5, e.files[0].path, e.files[0].sha256, e.files[0].bytes)
             for key, e in KNOWN_WEIGHTS.items()}
    assert table == W266
    for (method, model), entry in KNOWN_WEIGHTS.items():
        assert isinstance(entry, KnownWeights) and (entry.method, entry.model) == (method, model)
        assert len(entry.sha256) == 64 and len(entry.files) == 1 and entry.archive == (method == "spotiflow")
    assert KNOWN_WEIGHTS[("spotiflow", "smfish_3d")].url == (
        "https://github.com/weigertlab/spotiflow-models/releases/download/0.6.0/smfish_3d.zip")
    assert KNOWN_WEIGHTS[("piscis", "20230905")].url == (
        "https://huggingface.co/wniu/Piscis/resolve/9bdefc72cb519053c63fd2d7bff9d12db7bb394e/models/20230905.pt")
    assert {k: e.native_threshold for k, e in KNOWN_WEIGHTS.items() if k[0] == "piscis"} == {
        ("piscis", "20230905"): 0.5, ("piscis", "20251212"): 0.5}
    assert [round(KNOWN_WEIGHTS[("spotiflow", m)].native_threshold, 3) for m in
            ("synth_3d", "smfish_3d", "general", "hybiss")] == [0.3, 0.4, 0.5, 0.532]


def test_an_unknown_model_lists_the_known_models(cache):
    with pytest.raises(ValueError, match=r"known models: \['20230905', '20251212'\]"):
        resolve_weights("piscis", "latest")
    with pytest.raises(ValueError, match="known methods"):
        resolve_weights("starfish", "x")


# --- Cache location -------------------------------------------------------------------------------------

def test_the_cache_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("STARFINDER_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert weights_directory() == tmp_path / "xdg" / "starfinder" / "weights"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert weights_directory() == Path.home() / ".cache" / "starfinder" / "weights"
    monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(tmp_path / "override"))
    assert weights_directory() == tmp_path / "override"
    assert weights_directory(tmp_path / "explicit") == tmp_path / "explicit"


# --- Resolution and its errors --------------------------------------------------------------------------

def test_an_empty_cache_raises_missing_weights_naming_path_and_command(cache):
    with pytest.raises(MissingWeightsError) as raised:
        resolve_weights("spotiflow", "smfish_3d")
    message = str(raised.value)
    assert isinstance(raised.value, FileNotFoundError)
    assert str(cache / "spotiflow" / "smfish_3d" / "best.pt") in message
    assert "'starfinder weights fetch spotiflow smfish_3d'" in message


def test_fetch_writes_the_verified_copy_and_its_record(cache, piscis_fixture):
    folder = fetch_weights("piscis", "fixture")
    assert folder == cache / "piscis" / "fixture"
    assert (folder / "fixture.pt").read_bytes() == PAYLOAD
    record = json.loads((folder / RECORD_NAME).read_text())
    assert (record["method"], record["model"], record["url"], record["sha256"], record["bytes"]) == (
        "piscis", "fixture", piscis_fixture.url, sha256(PAYLOAD), len(PAYLOAD))
    assert record["files"] == {"fixture.pt": sha256(PAYLOAD)}
    assert sorted(p.name for p in folder.parent.iterdir()) == ["fixture"]  # no temporary files remain
    assert resolve_weights("piscis", "fixture") == folder
    assert weights_artifacts("piscis", "fixture", folder) == [{
        "name": "piscis/fixture", "path": str((folder / "fixture.pt").resolve()), "sha256": sha256(PAYLOAD),
        "source": piscis_fixture.url, "revision": "fixture"}]


def test_a_changed_byte_raises_the_hash_mismatch_with_both_hashes(cache, piscis_fixture):
    folder = fetch_weights("piscis", "fixture")
    changed = bytearray(PAYLOAD)
    changed[0] ^= 1
    (folder / "fixture.pt").write_bytes(bytes(changed))
    with pytest.raises(WeightsHashMismatchError) as raised:
        resolve_weights("piscis", "fixture")
    assert isinstance(raised.value, ValueError)
    assert sha256(bytes(changed)) in str(raised.value) and sha256(PAYLOAD) in str(raised.value)
    assert str(folder / "fixture.pt") in str(raised.value)
    # A changed copy is never overwritten by a new fetch.
    with pytest.raises(WeightsHashMismatchError):
        fetch_weights("piscis", "fixture")
    assert (folder / "fixture.pt").read_bytes() == bytes(changed)


def test_a_verified_copy_is_not_downloaded_again(cache, piscis_fixture):
    folder = fetch_weights("piscis", "fixture")
    Path(piscis_fixture.url.removeprefix("file://")).unlink()
    assert fetch_weights("piscis", "fixture") == folder


def test_a_wrong_download_installs_nothing(cache, piscis_fixture, monkeypatch):
    monkeypatch.setitem(KNOWN_WEIGHTS, ("piscis", "fixture"), replace(piscis_fixture, sha256="0" * 64))
    with pytest.raises(WeightsHashMismatchError, match="0" * 64):
        fetch_weights("piscis", "fixture")
    assert list((cache / "piscis").iterdir()) == []


def test_an_archive_is_checked_and_extracted(cache, archive_fixture, monkeypatch):
    folder = fetch_weights("spotiflow", "fixture_3d")
    assert folder == cache / "spotiflow" / "fixture_3d"
    assert sorted(p.name for p in folder.iterdir()) == ["best.pt", "config.yaml", RECORD_NAME]
    record = json.loads((folder / RECORD_NAME).read_text())
    assert record["md5"] == archive_fixture.md5
    assert record["files"] == {"best.pt": sha256(PAYLOAD), "config.yaml": sha256(b"is_3d: true\n")}
    assert not (cache / "spotiflow" / "__MACOSX").exists()
    other = cache.parent / "other"
    monkeypatch.setitem(KNOWN_WEIGHTS, ("spotiflow", "fixture_3d"), replace(archive_fixture, md5="f" * 32))
    with pytest.raises(WeightsHashMismatchError, match="MD5"):
        fetch_weights("spotiflow", "fixture_3d", directory=other)
    assert list((other / "spotiflow").iterdir()) == []


def test_the_directory_argument_overrides_the_environment(cache, piscis_fixture, tmp_path):
    folder = fetch_weights("piscis", "fixture", directory=tmp_path / "explicit")
    assert folder == tmp_path / "explicit" / "piscis" / "fixture"
    assert not cache.exists()
    with pytest.raises(MissingWeightsError, match="--dir"):
        resolve_weights("piscis", "fixture", directory=tmp_path / "elsewhere")


# --- Command line -------------------------------------------------------------------------------------

def test_the_weights_command_group(cache, piscis_fixture, capsys):
    assert main(["weights", "list"]) == 0
    listing = capsys.readouterr().out
    assert f"weights directory: {cache}" in listing
    assert all(f"{method}\t{model}\t" in listing for method, model in W266)
    assert "piscis\tfixture\t" in listing and "not fetched" in listing
    assert main(["weights", "verify"]) == 0
    assert "no local weights" in capsys.readouterr().out
    assert main(["weights", "fetch", "piscis", "fixture"]) == 0
    assert capsys.readouterr().out.strip() == str(cache / "piscis" / "fixture")
    assert main(["weights", "verify", "piscis", "fixture"]) == 0
    assert "piscis\tfixture\tverified" in capsys.readouterr().out
    assert main(["weights", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.rsplit("\t", 1)[1] for line in lines if line.startswith("piscis\tfixture\t")] == ["fetched"]
    (cache / "piscis" / "fixture" / "fixture.pt").write_bytes(b"changed")
    assert main(["weights", "verify"]) == 1
    assert "failed" in capsys.readouterr().out
    assert main(["weights", "verify", "piscis", "20251212"]) == 1
    assert "weights fetch piscis 20251212" in capsys.readouterr().out
    with pytest.raises(SystemExit) as raised:
        main(["weights", "verify", "piscis"])
    assert raised.value.code == 2
