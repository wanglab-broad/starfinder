"""Download and extraction of a weights file; imported only inside the functions that fetch.

Kept apart from ``starfinder._weights`` so that ``urllib`` and ``zipfile`` load
only when a model is fetched (``starfinder weights fetch``, fetch_weights).
"""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile


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
