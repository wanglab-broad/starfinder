"""Explicit weights download (starfinder weights fetch); never imported by a detection."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

from ._errors import WeightsHashMismatchError
from ._weights import RECORD_NAME, resolve_weights, sha256_file


def _download(url, target, chunk=1 << 22):
    """Stream url into target; return its size, SHA-256 and MD5."""
    sha256, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with urllib.request.urlopen(url, timeout=120) as response, open(target, "wb") as handle:
        while block := response.read(chunk):
            handle.write(block)
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return size, sha256.hexdigest(), md5.hexdigest()


def _extract(archive, destination):
    """Extract a zip without leaving destination; return the folder that holds the model files."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] == "__MACOSX" or PurePosixPath(member.filename).name == ".DS_Store":
                continue
            target = (root / Path(*parts)).resolve()
            if root not in target.parents:
                raise ValueError(f"archive member {member.filename!r} leaves the extraction folder")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
    entries = list(root.iterdir())
    # Archives hold one top folder named after the model, or the files themselves.
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else root


def _record(entry, folder):
    files = {p.relative_to(folder).as_posix(): sha256_file(p)
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != RECORD_NAME}
    return {"method": entry.method, "model": entry.model, "url": entry.url, "revision": entry.revision,
            "sha256": entry.sha256, "bytes": entry.bytes, "md5": entry.md5,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "files": files}


def fetch(entry, root):
    """Install entry under root/<method>/<model>/ after verifying the download; see fetch_weights."""
    parent = Path(root) / entry.method
    folder = parent / entry.model
    if folder.exists():
        # A verified copy is kept; anything else there is left for the user to inspect.
        resolve_weights(entry.method, entry.model, directory=root)
        if not (folder / RECORD_NAME).is_file():
            raise FileExistsError(f"{folder} exists without {RECORD_NAME}; remove it and fetch again")
        return folder
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{entry.model}.", suffix=".tmp", dir=parent))
    try:
        download = staging / "download"
        size, sha256, md5 = _download(entry.url, download)
        if size != entry.bytes or sha256 != entry.sha256:
            raise WeightsHashMismatchError(
                f"{entry.url}: downloaded {size} bytes with SHA-256 {sha256}, expected {entry.bytes} bytes with "
                f"SHA-256 {entry.sha256}")
        if entry.md5 is not None and md5 != entry.md5:
            raise WeightsHashMismatchError(f"{entry.url}: MD5 {md5}, expected the library's {entry.md5}")
        if entry.archive:
            extracted = _extract(download, staging / "extracted")
        else:
            extracted = staging / "extracted"
            extracted.mkdir()
            os.replace(download, extracted / entry.files[0].path)
        for item in entry.files:
            path = extracted / item.path
            if not path.is_file():
                raise WeightsHashMismatchError(f"{entry.url}: the download has no {item.path}")
            actual = sha256_file(path)
            if actual != item.sha256:
                raise WeightsHashMismatchError(f"{entry.url}: {item.path} has SHA-256 {actual}, expected {item.sha256}")
        record = _record(entry, extracted)
        (extracted / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
        os.replace(extracted, folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return resolve_weights(entry.method, entry.model, directory=root)
